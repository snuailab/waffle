version=$(python -c "from app import src; print(src.__version__)")
echo "Building version $version"
docker build --build-arg NO_CACHE_ARG=$(date +%s) --tag snuailab/waffle-app:$version --tag snuailab/waffle-app ./app
