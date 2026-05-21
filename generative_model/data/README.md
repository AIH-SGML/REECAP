### Example data

Because the original UK Biobank (UKBB) data used in this project is protected, this repository includes **example data** to demonstrate the pipeline without requiring UKBB access.

- **Images:**  
  Taken from the public [*Fundus Image 1000* dataset](https://www.kaggle.com/datasets/linchundan/fundusimage1000) by the Joint Shantou International Eye Centre (JSIEC).  
  The dataset is freely available on Kaggle and may be used for non-commercial research purposes.  
  A small subset is included in `imgs/` for illustration.

- **Embeddings and phenotypes:**  
  The file `example_embeddings.h5ad` contains randomly generated (synthetic) phenotype values aligned with the example images.  
  It mimics the structure of the real UKBB embeddings but contains **no real or identifiable data**.

You can use these files to test data loading, embedding, and fine-tuning pipelines.  
To run on real UKBB data, simply replace the example paths with your authorized data locations.