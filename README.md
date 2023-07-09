# BrazilianSoccerEDA

Analysis with data of [Brazilian Soccer](https://github.com/IgorMichels/BrazilianSoccerData).

## Extract data

In order to execute all these analysis, you must follow these steps
```shell
# inside this directory, create a folder with data and clone BrazilianSoccerData repository
mkdir data
cd data
git clone -n --depth=1 --filter=tree:0 git@github.com:IgorMichels/BrazilianSoccerData.git
cd BrazilianSoccerData
git sparse-checkout set --no-cone results
git checkout
```
