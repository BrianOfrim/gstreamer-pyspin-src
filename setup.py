import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gst-pyspin-src",
    version="0.0.1",
    author="Brian Ofrim",
    author_email="bofrim@ualberta.ca",
    description="GStreamer source plug-in for the FLIR Spinnaker Python SDK (PySpin)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BrianOfrim/gstreamer-pyspin-src",
    packages=setuptools.find_packages(),
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu",
    ],
    python_requires=">=3.6",
)
