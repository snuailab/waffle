mkdir -p docs/tutorials
cp -r tutorials/* docs/tutorials/

git clone https://github.com/snuailab/waffle_hub
git clone https://github.com/snuailab/waffle_menu

pip install -U waffle_hub
pip install -U waffle_menu

mkdocs gh-deploy --clean