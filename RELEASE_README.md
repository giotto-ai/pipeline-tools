# Release Readme

## Who can do it

Maintainers or above can create releases.

Pleas make sure to properly use the version numbers according to [these standards](https://semver.org/#:~:text=A%20normal%20version%20number%20MUST,0%20%2D%3E%201.11.0.).

## How to do it

These are the steps to follow to make a new release of this package, in this order. Of course, you have to make sure that all CI tests are passing that you have also manually tested that everything works.

### Step 1: 

Change the version in the [pyproject.toml](./pyproject.toml) file

### Step 2:

Draft a new release [here](https://github.com/giotto-ai/pipeline-tools/releases/new). Feel free to fill it in with the aoutomated button. 

In the release view, create a new tag called `vX.y.Z`, with `X`, `Y`, and `Z` the major, minor and patch version.
Leave the main ranch as target.

Once you are done, please make sure to click on the "Save draft". Do not publish yet!

### Step 3:

Run this [action job](https://github.com/giotto-ai/pipeline-tools/actions/workflows/python-publish.yml) manually, by clicking the "Run workflow" button on the top right. 
This job, if successful, will deploy the package to `pypi` directly. You can check it online and `pip install` it.

If the job fails, it means that there are probably issues in packaging and building the project: analyse the CI and fix all that is needed.

### Step 4:

Once published on pypi via the action job of the previous step, complete, if needed, the new release you saved as draft at step 2. 

Once you are happy with the text, publish it (click the green button)!

