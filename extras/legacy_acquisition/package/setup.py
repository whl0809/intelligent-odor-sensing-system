from setuptools import Extension, setup


setup(
    ext_modules=[
        Extension(
            "enose._bme69x",
            sources=[
                "src/enose/_bme69xmodule.c",
                "src/enose/_vendor/bme690/bme69x.c",
            ],
            include_dirs=["src/enose/_vendor/bme690"],
        )
    ]
)
