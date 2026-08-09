#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <string.h>

#include "bme69x.h"

#define BOSCH_SENSORAPI_VERSION "1.1.0"

typedef struct
{
    PyObject_HEAD
    PyObject *bus;
    PyObject *sleep_fn;
    uint8_t address;
    int initialized;
    struct bme69x_dev device;
    struct bme69x_conf configuration;
    struct bme69x_heatr_conf heater;
} BME690Driver;

static PyObject *SensorAPIError;
static PyObject *NoDataError;

static BME69X_INTF_RET_TYPE bus_read(
    uint8_t reg_addr,
    uint8_t *reg_data,
    uint32_t length,
    void *intf_ptr)
{
    BME690Driver *self = (BME690Driver *)intf_ptr;
    PyObject *result;
    Py_buffer buffer;

    result = PyObject_CallMethod(
        self->bus,
        "read_register",
        "iii",
        (int)self->address,
        (int)reg_addr,
        (int)length);
    if (result == NULL)
    {
        return BME69X_E_COM_FAIL;
    }

    if (PyObject_GetBuffer(result, &buffer, PyBUF_SIMPLE) < 0)
    {
        Py_DECREF(result);
        return BME69X_E_COM_FAIL;
    }

    if (buffer.len != (Py_ssize_t)length)
    {
        PyErr_Format(
            PyExc_ValueError,
            "BME690 register read returned %zd bytes; expected %u",
            buffer.len,
            (unsigned int)length);
        PyBuffer_Release(&buffer);
        Py_DECREF(result);
        return BME69X_E_COM_FAIL;
    }

    memcpy(reg_data, buffer.buf, length);
    PyBuffer_Release(&buffer);
    Py_DECREF(result);
    return BME69X_INTF_RET_SUCCESS;
}

static BME69X_INTF_RET_TYPE bus_write(
    uint8_t reg_addr,
    const uint8_t *reg_data,
    uint32_t length,
    void *intf_ptr)
{
    BME690Driver *self = (BME690Driver *)intf_ptr;
    PyObject *payload;
    PyObject *result;
    char *payload_data;

    payload = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)length + 1);
    if (payload == NULL)
    {
        return BME69X_E_COM_FAIL;
    }

    payload_data = PyBytes_AS_STRING(payload);
    payload_data[0] = (char)reg_addr;
    memcpy(payload_data + 1, reg_data, length);

    result = PyObject_CallMethod(
        self->bus,
        "write",
        "iO",
        (int)self->address,
        payload);
    Py_DECREF(payload);
    if (result == NULL)
    {
        return BME69X_E_COM_FAIL;
    }

    Py_DECREF(result);
    return BME69X_INTF_RET_SUCCESS;
}

static void delay_us(uint32_t period, void *intf_ptr)
{
    BME690Driver *self = (BME690Driver *)intf_ptr;
    PyObject *result = PyObject_CallFunction(
        self->sleep_fn,
        "d",
        (double)period / 1000000.0);

    Py_XDECREF(result);
}

static int set_api_error(const char *operation, int8_t result)
{
    PyObject *arguments;
    PyObject *exception;

    if (PyErr_Occurred())
    {
        return 0;
    }

    arguments = Py_BuildValue("(si)", operation, (int)result);
    if (arguments == NULL)
    {
        return 0;
    }

    exception = PyObject_CallObject(SensorAPIError, arguments);
    Py_DECREF(arguments);
    if (exception == NULL)
    {
        return 0;
    }

    PyErr_SetObject(SensorAPIError, exception);
    Py_DECREF(exception);
    return 0;
}

static int require_api_ok(const char *operation, int8_t result)
{
    if (result == BME69X_OK && !PyErr_Occurred())
    {
        return 1;
    }

    return set_api_error(operation, result);
}

static PyObject *driver_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    (void)args;
    (void)kwargs;

    BME690Driver *self = (BME690Driver *)type->tp_alloc(type, 0);

    if (self != NULL)
    {
        self->bus = NULL;
        self->sleep_fn = NULL;
        self->address = BME69X_I2C_ADDR_LOW;
        self->initialized = 0;
        memset(&self->device, 0, sizeof(self->device));
        memset(&self->configuration, 0, sizeof(self->configuration));
        memset(&self->heater, 0, sizeof(self->heater));
    }

    return (PyObject *)self;
}

static int driver_init(BME690Driver *self, PyObject *args, PyObject *kwargs)
{
    static char *keywords[] = { "bus", "sleep_fn", "address", NULL };
    PyObject *bus;
    PyObject *sleep_fn;
    int address = BME69X_I2C_ADDR_LOW;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OO|i:Driver",
            keywords,
            &bus,
            &sleep_fn,
            &address))
    {
        return -1;
    }

    if (!PyCallable_Check(sleep_fn))
    {
        PyErr_SetString(PyExc_TypeError, "sleep_fn must be callable");
        return -1;
    }
    if (address < 0x03 || address > 0x77)
    {
        PyErr_SetString(PyExc_ValueError, "address must be a valid 7-bit I2C address");
        return -1;
    }

    Py_INCREF(bus);
    Py_INCREF(sleep_fn);
    Py_XSETREF(self->bus, bus);
    Py_XSETREF(self->sleep_fn, sleep_fn);
    self->address = (uint8_t)address;
    self->initialized = 0;

    memset(&self->device, 0, sizeof(self->device));
    self->device.read = bus_read;
    self->device.write = bus_write;
    self->device.delay_us = delay_us;
    self->device.intf = BME69X_I2C_INTF;
    self->device.intf_ptr = self;
    self->device.amb_temp = 25;
    return 0;
}

static void driver_dealloc(BME690Driver *self)
{
    Py_XDECREF(self->bus);
    Py_XDECREF(self->sleep_fn);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *driver_initialize(BME690Driver *self, PyObject *args)
{
    int heater_temperature_c;
    int heater_duration_ms;
    int8_t result;

    if (!PyArg_ParseTuple(
            args,
            "ii:initialize",
            &heater_temperature_c,
            &heater_duration_ms))
    {
        return NULL;
    }
    if (heater_temperature_c < 0 || heater_temperature_c > 400)
    {
        PyErr_SetString(
            PyExc_ValueError,
            "heater_temperature_c must be between 0 and 400");
        return NULL;
    }
    if (heater_duration_ms < 0 || heater_duration_ms > UINT16_MAX)
    {
        PyErr_SetString(
            PyExc_ValueError,
            "heater_duration_ms must be between 0 and 65535");
        return NULL;
    }

    self->initialized = 0;
    result = bme69x_init(&self->device);
    if (!require_api_ok("bme69x_init", result))
    {
        return NULL;
    }

    if (self->device.variant_id != BME690_VARIANT_GAS_HIGH)
    {
        return Py_BuildValue(
            "(II)",
            (unsigned int)self->device.chip_id,
            (unsigned int)self->device.variant_id);
    }

    memset(&self->configuration, 0, sizeof(self->configuration));
    self->configuration.filter = BME69X_FILTER_OFF;
    self->configuration.odr = BME69X_ODR_NONE;
    self->configuration.os_hum = BME69X_OS_16X;
    self->configuration.os_pres = BME69X_OS_16X;
    self->configuration.os_temp = BME69X_OS_16X;
    result = bme69x_set_conf(&self->configuration, &self->device);
    if (!require_api_ok("bme69x_set_conf", result))
    {
        return NULL;
    }

    memset(&self->heater, 0, sizeof(self->heater));
    self->heater.enable = BME69X_ENABLE;
    self->heater.heatr_temp = (uint16_t)heater_temperature_c;
    self->heater.heatr_dur = (uint16_t)heater_duration_ms;
    result = bme69x_set_heatr_conf(
        BME69X_FORCED_MODE,
        &self->heater,
        &self->device);
    if (!require_api_ok("bme69x_set_heatr_conf", result))
    {
        return NULL;
    }

    self->initialized = 1;
    return Py_BuildValue(
        "(II)",
        (unsigned int)self->device.chip_id,
        (unsigned int)self->device.variant_id);
}

static PyObject *driver_read(BME690Driver *self, PyObject *Py_UNUSED(ignored))
{
    struct bme69x_data data;
    uint32_t delay_period;
    uint8_t field_count = 0;
    int8_t result;
    PyObject *gas_valid;
    PyObject *heater_stable;
    PyObject *sample;

    if (!self->initialized)
    {
        PyErr_SetString(SensorAPIError, "BME690 SensorAPI is not initialized");
        return NULL;
    }

    memset(&data, 0, sizeof(data));
    result = bme69x_set_op_mode(BME69X_FORCED_MODE, &self->device);
    if (!require_api_ok("bme69x_set_op_mode", result))
    {
        return NULL;
    }

    delay_period = bme69x_get_meas_dur(
        BME69X_FORCED_MODE,
        &self->configuration,
        &self->device);
    delay_period += (uint32_t)self->heater.heatr_dur * 1000U;
    self->device.delay_us(delay_period, self->device.intf_ptr);
    if (PyErr_Occurred())
    {
        return NULL;
    }

    result = bme69x_get_data(
        BME69X_FORCED_MODE,
        &data,
        &field_count,
        &self->device);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    if (result == BME69X_W_NO_NEW_DATA || field_count == 0)
    {
        PyErr_SetString(NoDataError, "BME690 returned no new forced-mode sample");
        return NULL;
    }
    if (!require_api_ok("bme69x_get_data", result))
    {
        return NULL;
    }

    gas_valid = PyBool_FromLong((data.status & BME69X_GASM_VALID_MSK) != 0);
    heater_stable = PyBool_FromLong((data.status & BME69X_HEAT_STAB_MSK) != 0);
    if (gas_valid == NULL || heater_stable == NULL)
    {
        Py_XDECREF(gas_valid);
        Py_XDECREF(heater_stable);
        return NULL;
    }

    sample = Py_BuildValue(
        "(ddddOO)",
        (double)data.temperature,
        (double)data.humidity,
        (double)data.pressure,
        (double)data.gas_resistance,
        gas_valid,
        heater_stable);
    Py_DECREF(gas_valid);
    Py_DECREF(heater_stable);
    return sample;
}

static PyMethodDef driver_methods[] = {
    {
        "initialize",
        (PyCFunction)driver_initialize,
        METH_VARARGS,
        PyDoc_STR("Initialize and configure Bosch SensorAPI forced mode.")
    },
    {
        "read",
        (PyCFunction)driver_read,
        METH_NOARGS,
        PyDoc_STR("Trigger and return one compensated forced-mode sample.")
    },
    { NULL, NULL, 0, NULL }
};

static PyTypeObject DriverType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "enose._bme69x.Driver",
    .tp_basicsize = sizeof(BME690Driver),
    .tp_dealloc = (destructor)driver_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("Small bridge to Bosch BME690 SensorAPI."),
    .tp_methods = driver_methods,
    .tp_init = (initproc)driver_init,
    .tp_new = driver_new,
};

static PyModuleDef bme69x_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_bme69x",
    .m_doc = "Native bridge to Bosch Sensortec BME690 SensorAPI.",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__bme69x(void)
{
    PyObject *module;

    if (PyType_Ready(&DriverType) < 0)
    {
        return NULL;
    }

    module = PyModule_Create(&bme69x_module);
    if (module == NULL)
    {
        return NULL;
    }

    SensorAPIError = PyErr_NewException(
        "enose._bme69x.SensorAPIError",
        PyExc_RuntimeError,
        NULL);
    NoDataError = PyErr_NewException(
        "enose._bme69x.NoDataError",
        SensorAPIError,
        NULL);
    if (SensorAPIError == NULL || NoDataError == NULL)
    {
        Py_XDECREF(SensorAPIError);
        Py_XDECREF(NoDataError);
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(&DriverType);
    if (PyModule_AddObject(module, "Driver", (PyObject *)&DriverType) < 0)
    {
        Py_DECREF(&DriverType);
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(SensorAPIError);
    if (PyModule_AddObject(module, "SensorAPIError", SensorAPIError) < 0)
    {
        Py_DECREF(SensorAPIError);
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(NoDataError);
    if (PyModule_AddObject(module, "NoDataError", NoDataError) < 0)
    {
        Py_DECREF(NoDataError);
        Py_DECREF(module);
        return NULL;
    }

    if (PyModule_AddStringConstant(
            module,
            "SENSORAPI_VERSION",
            BOSCH_SENSORAPI_VERSION) < 0)
    {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
