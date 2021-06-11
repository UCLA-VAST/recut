Recut replaces the needed python functionality (most notably the soma inferenece/segmentation) that was in the final versions of the master branch and the pipeline zip files. mcp3d will stay private to our collaborators because of legal issues, whereas recut is being open sourced and will optionally reference mcp3d.

# MCP3D goals
- [X] Recut references the mcp3d master branch repo remotely for image reading (not writing) of tif and .ims/hdf5 files. 
- [X] Recut tests this functionality in the automated test suite
- [ ] All existing unit tests establishing the above read goal should pass from the mcp3d_unit_tests executable
- [ ] Any other tests that do not pass can be DISABLED in gtest. 
- [ ] If any other tests in mcp3d_unit_tests seem practical for preventing regression in needed behavior, they can also be fixed_
- [ ] Adding support for more file formats (removing FusionSticher requirments).

# Python pipeline goals
- [ ] removing the python section of mcp3D.
- [ ] Those who have access to mcp3d on GitHub should have access to the recut repo as well so that they can have access to updated documents.
- [ ] Keeping the documentations up to date.
- [ ] testing the possibility of running the pipeline in windoes subsystem for Linux (WSL).

# ML model
- [ ] trainig the model for rabies labeling data.
- [ ] tuning hyperparameters.
- [ ] testing 3D convolution operation instead of current 2D convolution.

# Pipeline release
- [ ] documenting pipeline release process.

# Current working pipeline
![](images/pipeline.png?raw=true)

# Future pipeline
![](images/future-pipeline.png?raw=true)
