# Spectral Semantic Calibration

Core implementation of the Spectral Semantic Calibration (SSC) method.

SSC compares token predictions from an original chest X-ray and a spatially downsampled and upsampled view. It applies bounded multiplicative calibration within the Top-$K$ candidates selected from the original-view prediction distribution.

## Repository Contents

`inference_ssc.py` contains the implementation of SSC.

## Experimental Settings

The SSC parameters used in the paper are:

| Parameter | Value |
| --- | ---: |
| Downsampling factor `s` | 8 |
| Modulation strength `alpha` | 1.0 |
| Candidate pool size `K` | 50 |
| Stability constant `epsilon` | 1e-12 |
| Maximum number of new tokens | 512 |

## Acknowledgments

SSC uses LLaVA-Rad and R2GenGPT as backbone models. When using their code or models, please follow the licenses and citation requirements of the upstream repositories.# Spectral-Semantic-Calibration
