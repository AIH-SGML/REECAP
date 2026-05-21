### Example data

Because the original UK Biobank (UKBB) data used in this project is protected, this repository includes **example data** to demonstrate the pipeline without requiring UKBB access.

- **Starting weights for RETFound**
First download the weights for the RETFound model from the original RETFound\_mae\_natureCFP (ViT large)'RETFound_mae_natureCFP.pth' here.

- **Images:**  
  Taken from the public [*Fundus Image 1000* dataset](https://www.kaggle.com/datasets/linchundan/fundusimage1000) by the Joint Shantou International Eye Centre (JSIEC).  
  The dataset is freely available on Kaggle and may be used for non-commercial research purposes.  
  A small subset is included in `./imgs/` for illustration.
  Here we provide the square images what pass the quality filter. 
  If you need to perform the preprocessing and image filtering yourself, in the REECAP we used the EyeQ library, adjusted to crop the image if it is not square and set the threshold for "Reject" parameter to 0.8 (see https://github.com/AIH-SGML/EyeQ.git).

- **Embeddings and phenotypes:**  
  The file `path_table.csv` contains randomly generated (synthetic) phenotype values aligned with the example images.  