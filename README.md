Our aim to understand the financial conditions of company fundraising goals.

Topic: Building-Machine-Learning-Pipeline-on-Startup-Acquisition
Objective:
The objective of the project is to predict whether a startup which is currently Operating, IPO, Acquired, or closed.This problem will be solved through a Supervised Machine Learning approach by training a model based on the history of startups which were either acquired or closed.

Data:
Link to raw data(Huge JSON and Excel fiel):
    https://drive.google.com/file/d/1tWYkHYHm2HoiCajZ49Cs1K7sklWTdAbV/view
Summary:
The data contains industry trends, investment insights and individual company information. Since the data was acquired on a trial basis, it only contains information about companies. After training the model, we predict whether startups still operating, IPO, acquired, or closed.

Data types:
There are 44 columns out of which will be used as features. The rest provide more information about the data, but will not be used for model training (like normalized name, entity id, short description etc.)

Data Cleaning:
In data cleaning we are will remove the inappropriate & unncessary information from raw data(main data).Hence, after which we’ll perform data cleaning involving following steps:

Checking the percentage of NaN(null values) values present in each feature
Removing duplicates
Dropping columns which have NaN values.
Remove unnecessary and corrupted data.
Data Labelling.
kaggle kernels output harsh9759/acquisition-status-self-scored-python-companies -p /path/to/dest
