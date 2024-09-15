# Parameter identification

The parameter calibration is done using the [JuliaSimModelOptimizer](https://help.juliahub.com/jsmo/stable/) library, a component of the [JuliaSim](https://juliahub.com/products/juliasim) suite developed by [JuliaHub Inc](https://juliahub.com). For further information, I recommend the following [video](https://www.youtube.com/watch?v=TkmpICaFDrM). The installation of this package is detailled on their website.

The main Julia file is 'calibration.jl'.
A local Julia environment is required as JuliaSimModelOptimizer is compatible with old version of ModelingToolkit. For this reason, other ways of defining the model is followed in file 'model_st_tanh.jl' for instance. Other models can be used in the same way.

## Other thoughts

Lot of possible analysis are made possible with this package. The youtube chanel [JuliaHubInc](https://www.youtube.com/@JuliaHubInc/videos) is recommended to have more insights on what is possible to do in Julia. Notably, the series of videos 'system identification with Julia' is recommended here, and available [here](https://www.youtube.com/watch?v=RnuHqkP4QTw).
