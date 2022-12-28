# ImgAnalysis.jl

## #1 Installation

```
julia> ]
pkg> registry add https://github.com/FemtoPhysics/FemtoPhysics-Registry.git#main
pkg> add ImgAnalysis
```

Optionally, you can also install

```
pkg> add FileIO ImageIO ImageShow Plots
```

## #2 Usage

### #2.1 Preprocess the image step by step

```julia
import FileIO, ImageIO, ImageShow, Plots
import ImgAnalysis: rgb_to_gray, Corrector, correction, leveling
```

1. Import the image and convert it to a grayscale matrix
```julia
imgGray = rgb_to_gray(FileIO.load("<img path here>"))
```

2. Correct the shading effect by using Legendre polynomials
```julia
# Polynomial degree up to 2 or 3 is generally enough
height_order = 3
width_order  = 3

corrector = Corrector(imgGray, height_order, width_order)
imgGrayCorrect = correction(corrector)
```

3. Background subtraction by data leveling
```julia
imgGrayCorrectLevel = leveling(imgGrayCorrect)
```

4. Demo the result
```julia
Plots.heatmap(imgGrayCorrectLevel; color=:grays, aspect_ratio=:equal, yflip=true)
```

### #2.2 Directly preprocess the image and save to a new image file

```julia
import ImgAnalysis: preprocess
preprocess(
    "< src. img. >.jpg";
    ifsave=true, fname="< des. img. >.jpg",
    height_order=3, width_order=3
)
```
