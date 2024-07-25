echo [$(date)] : "START"

echo [$(date)] : "Creating a new conda environment:"
echo [$(date)] : conda create --prefix ./env python -y

echo [$(date)] : "Activating the environment:"
echo [$(date)] : conda activate ./env 

echo [$(date)] : "Installing the dev dependencies and requirements:"
echo [$(date)] : pip install -r requirements_dev.txt

echo [$(date)] : "END"