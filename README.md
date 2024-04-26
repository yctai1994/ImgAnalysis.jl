# ImgAnalysis.jl

## # 1 Installation

```julia
julia> import Pkg
julia> Pkg.add(url="https://github.com/yctai1994/ImgAnalysis.jl.git")
```

Optionally, you can also install

```julia
julia> Pkg.add("CairoMakie")
```

## # 2 Preprocess the image

### # Directly preprocess the image and save it to a new image file

```julia
using ImgAnalysis
preprocess(
    "$fname.jpg";
    ifsave=true, fname="$fname-processed.jpg",
    height_order=3, width_order=3
)
```

## # 3 Clustering Pixels and Find Their Area

```julia
import FileIO, ImageIO, CairoMakie, DelimitedFiles
using ImgAnalysis
```

1. Import the preprocessed image and prepare the `classifier`.
```julia
fname  = "<img path here>" # no file extension
imgSrc = Float32.(FileIO.load("$fname.jpg"))
classifier = ImgClassifier(imgSrc;)
```

2. Assign hyperparameters and compute the kernel (Gram) matrix. (The comments  on the right-hand side are the alternatives if you want to avoid typing the full names of all arguments.)
```julia
set_params!(classifier;
    cluster_num      = 4,   # K  = 4,
    height_scaler    = 1.5, # γH = 1.5,
    width_scaler     = 1.5, # γW = 1.5,
    grayscale_scaler = 6.0  # γG = 6.0
)
```

3. Solve the clustering.
```julia
solve!(classifier)
```

4. Demo the result (in `ImgAnalysis.jl.git/tmp`).
```julia
demo(imgSrc, classifier.result;
     ifsave=false, fname="$fname-compare")
```

5. You can find all the linked-pixel areas upon your interests using
```julia
# target_label::Int is the label of
# clustering, e.g., 1, 2, 3.
filtered_labels = count_area(classifier.result, target_label)

argmax(x -> x[3], filtered_labels)
```

, or you can save the clustering result to a `.txt` file  (in `ImgAnalysis.jl.git/tmp`).
```julia
params = classifier.params
save_result("$fname.txt", classifier.result;
            K=params.K, P=params.Pinit,
            γH=params.γH, γW=params.γW, γG=params.γG)
```
