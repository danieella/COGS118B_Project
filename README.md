# Bird Image Classification Project

In this project, we explored how principal components in CLIP embeddings relate to background information of 200+ bird images, and whether removing these components affects bird species classification. Using images from a Bird Species Classification dataset, we performed PCA on CLIP features and gradually removed the top components to see how accuracy changed for each species.

**Tools:** Python, CLIP, Scikit-learn, Matplotlib, Jupyter Notebook

**My Contributions:**
  - Collaborated with teammates during project work sessions to perform PCA analysis on CLIP embeddings
  - Refined the research hypothesis to focus on the relationship between principal components and background information
  - Collaborated to develop a visualization illustrating how classification accuracy changes as components are removed

**Results:** We found that removing principal components affects classification performance differently across bird species. Removing the top 3 PCs had no impact on accuracy, which suggests those components mostly capture background information irrelevant to classification. There is a spot around the 4th PC where dimensionality reduction is most beneficial, and some bird species proved easier to classify than others. Given more time, we would have explored alternative methods beyond PCA and experiment with removing additional components to better understand their effect on classification.

## Project Notebook
You can view the full analysis here:

[View Full Notebook](118bcode_3.py)



