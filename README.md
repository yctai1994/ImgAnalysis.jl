# ImgAnalysis.jl

## # 1 Installation

```
julia> ]
pkg> registry add https://github.com/FemtoPhysics/FemtoPhysics-Registry.git#main
pkg> add ImgAnalysis
```

Optionally, you can also install

```
pkg> add FileIO ImageIO ImageShow Plots
```

## # 2 Usage

### # 2.1 Preprocess the image step by step

```julia
import FileIO, ImageIO, ImageShow, Plots
import ImgAnalysis: rgb_to_gray, Corrector, correction, leveling
```

1. Import the image and convert it to a grayscale matrix.
```julia
fname   = "<img path here>" # no file extension
imgGray = Float32.(rgb_to_gray(FileIO.load(fname)))
```

2. Correct the shading effect by using Legendre polynomials.
```julia
# A Polynomial degree up to 2 or 3 is generally enough.
height_order = 3
width_order  = 3

corrector = Corrector(imgGray, height_order, width_order)
imgGrayCorrect = correction(corrector)
```

3. Background subtraction by data leveling.
```julia
imgGrayCorrectLevel = leveling(imgGrayCorrect)
```

4. Demo the result.
```julia
Plots.heatmap(imgGrayCorrectLevel; color=:grays, aspect_ratio=:equal, yflip=true)
```

### # 2.2 Directly preprocess the image and save it to a new image file

```julia
import ImgAnalysis: preprocess
preprocess(
    "$fname.jpg";
    ifsave=true, fname="$fname-processed.jpg",
    height_order=3, width_order=3
)
```

## # 3 Clustering Pixels and Find Their Area

```julia
import FileIO, ImageIO, Plots, DelimitedFiles
import ImgAnalysis: encoding, kernel!, kmeanspp!, iterate!, count_area
```

1. Import the preprocessed image.
```julia
fname  = "<img path here>" # no file extension
imgSrc = Float32.(FileIO.load("$fname.jpg"))
```

2. Assign hyperparameters using the following steps.
```julia
imgN  = length(imgSrc)
imgK  = 3              # number of clustering
imgP  = -1.0           # init. power value of power mean
imgγH = 1.5            # height-similarity scale
imgγW = 1.5            # width-similarity scale
imgγG = 6.0            # grayscale-similarity scale
```

3. Encode the source image into a 3D equidistant space.
```julia
imgDat = encoding(imgSrc)
```

4. Compute the kernel (Gram) matrix.
```julia
imgKernel = kernel!(
    Matrix{Float64}(undef, imgN, imgN), imgDat;
    γH=imgγH, γW=imgγW, γG=imgγG
)
```

5. Pre-allocate memory buffers to have efficient computation.
```julia
imgDist2Means = Matrix{Float64}(undef, imgN, imgK)
imgBufferNK   = similar(imgDist2Means)
imgWeights    = Matrix{Float64}(undef, imgN, imgK)
imgClusterVol = Matrix{Float64}(undef, 1, imgK)
imgBufferN1   = Vector{Float64}(undef, imgN)
imgResult     = similar(imgSrc, Int)
```

6. Initialize the clustering.
```julia
kmeanspp!(imgResult, imgWeights, imgDist2Means, imgBufferNK, imgBufferN1, imgKernel, imgP)
```

7. Update the clustering.
```julia
iterate!(
    imgResult, imgWeights, imgDist2Means,
    imgBufferNK, imgBufferN1, imgClusterVol,
    imgKernel, imgP * 1.04
)
```

8. Demo the result (in `ImgAnalysis.jl.git/tmp`).
```julia
demo(imgSrc, imgResult; ifsave=false, fname="$fname-compare")
```

9. You can find all the linked-pixel areas upon your interests using
```julia
# target_label::Int is the label of
# clustering, e.g., 1, 2, 3.
filtered_labels = count_area(imgResult, target_label)

argmax(x -> x[3], filtered_labels)
```

, or you can save the clustering result to a `.txt` file  (in `ImgAnalysis.jl.git/tmp`).
```julia
save_result("$fname.txt", imgResult; K=imgK, P=imgP, γH=imgγH, γW=imgγW, γG=imgγG)
```
