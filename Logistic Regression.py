{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "___\n",
    "# Logistic Regression Project - Solutions\n",
    "\n",
    "In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.\n",
    "\n",
    "This data set contains the following features:\n",
    "\n",
    "* 'Daily Time Spent on Site': consumer time on site in minutes\n",
    "* 'Age': cutomer age in years\n",
    "* 'Area Income': Avg. Income of geographical area of consumer\n",
    "* 'Daily Internet Usage': Avg. minutes a day consumer is on the internet\n",
    "* 'Ad Topic Line': Headline of the advertisement\n",
    "* 'City': City of consumer\n",
    "* 'Male': Whether or not consumer was male\n",
    "* 'Country': Country of consumer\n",
    "* 'Timestamp': Time at which consumer clicked on Ad or closed window\n",
    "* 'Clicked on Ad': 0 or 1 indicated clicking on Ad\n",
    "\n",
    "## Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "ad_data = pd.read_csv('advertising.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Daily Time Spent on Site</th>\n",
       "      <th>Age</th>\n",
       "      <th>Area Income</th>\n",
       "      <th>Daily Internet Usage</th>\n",
       "      <th>Ad Topic Line</th>\n",
       "      <th>City</th>\n",
       "      <th>Male</th>\n",
       "      <th>Country</th>\n",
       "      <th>Timestamp</th>\n",
       "      <th>Clicked on Ad</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>68.95</td>\n",
       "      <td>35</td>\n",
       "      <td>61833.90</td>\n",
       "      <td>256.09</td>\n",
       "      <td>Cloned 5thgeneration orchestration</td>\n",
       "      <td>Wrightburgh</td>\n",
       "      <td>0</td>\n",
       "      <td>Tunisia</td>\n",
       "      <td>2016-03-27 00:53:11</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>80.23</td>\n",
       "      <td>31</td>\n",
       "      <td>68441.85</td>\n",
       "      <td>193.77</td>\n",
       "      <td>Monitored national standardization</td>\n",
       "      <td>West Jodi</td>\n",
       "      <td>1</td>\n",
       "      <td>Nauru</td>\n",
       "      <td>2016-04-04 01:39:02</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>69.47</td>\n",
       "      <td>26</td>\n",
       "      <td>59785.94</td>\n",
       "      <td>236.50</td>\n",
       "      <td>Organic bottom-line service-desk</td>\n",
       "      <td>Davidton</td>\n",
       "      <td>0</td>\n",
       "      <td>San Marino</td>\n",
       "      <td>2016-03-13 20:35:42</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>74.15</td>\n",
       "      <td>29</td>\n",
       "      <td>54806.18</td>\n",
       "      <td>245.89</td>\n",
       "      <td>Triple-buffered reciprocal time-frame</td>\n",
       "      <td>West Terrifurt</td>\n",
       "      <td>1</td>\n",
       "      <td>Italy</td>\n",
       "      <td>2016-01-10 02:31:19</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>68.37</td>\n",
       "      <td>35</td>\n",
       "      <td>73889.99</td>\n",
       "      <td>225.58</td>\n",
       "      <td>Robust logistical utilization</td>\n",
       "      <td>South Manuel</td>\n",
       "      <td>0</td>\n",
       "      <td>Iceland</td>\n",
       "      <td>2016-06-03 03:36:18</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Daily Time Spent on Site  Age  Area Income  Daily Internet Usage  \\\n",
       "0                     68.95   35     61833.90                256.09   \n",
       "1                     80.23   31     68441.85                193.77   \n",
       "2                     69.47   26     59785.94                236.50   \n",
       "3                     74.15   29     54806.18                245.89   \n",
       "4                     68.37   35     73889.99                225.58   \n",
       "\n",
       "                           Ad Topic Line            City  Male     Country  \\\n",
       "0     Cloned 5thgeneration orchestration     Wrightburgh     0     Tunisia   \n",
       "1     Monitored national standardization       West Jodi     1       Nauru   \n",
       "2       Organic bottom-line service-desk        Davidton     0  San Marino   \n",
       "3  Triple-buffered reciprocal time-frame  West Terrifurt     1       Italy   \n",
       "4          Robust logistical utilization    South Manuel     0     Iceland   \n",
       "\n",
       "             Timestamp  Clicked on Ad  \n",
       "0  2016-03-27 00:53:11              0  \n",
       "1  2016-04-04 01:39:02              0  \n",
       "2  2016-03-13 20:35:42              0  \n",
       "3  2016-01-10 02:31:19              0  \n",
       "4  2016-06-03 03:36:18              0  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ad_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1000 entries, 0 to 999\n",
      "Data columns (total 10 columns):\n",
      " #   Column                    Non-Null Count  Dtype  \n",
      "---  ------                    --------------  -----  \n",
      " 0   Daily Time Spent on Site  1000 non-null   float64\n",
      " 1   Age                       1000 non-null   int64  \n",
      " 2   Area Income               1000 non-null   float64\n",
      " 3   Daily Internet Usage      1000 non-null   float64\n",
      " 4   Ad Topic Line             1000 non-null   object \n",
      " 5   City                      1000 non-null   object \n",
      " 6   Male                      1000 non-null   int64  \n",
      " 7   Country                   1000 non-null   object \n",
      " 8   Timestamp                 1000 non-null   object \n",
      " 9   Clicked on Ad             1000 non-null   int64  \n",
      "dtypes: float64(3), int64(3), object(4)\n",
      "memory usage: 78.2+ KB\n"
     ]
    }
   ],
   "source": [
    "ad_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Daily Time Spent on Site</th>\n",
       "      <th>Age</th>\n",
       "      <th>Area Income</th>\n",
       "      <th>Daily Internet Usage</th>\n",
       "      <th>Male</th>\n",
       "      <th>Clicked on Ad</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1000.000000</td>\n",
       "      <td>1000.000000</td>\n",
       "      <td>1000.000000</td>\n",
       "      <td>1000.000000</td>\n",
       "      <td>1000.000000</td>\n",
       "      <td>1000.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>65.000200</td>\n",
       "      <td>36.009000</td>\n",
       "      <td>55000.000080</td>\n",
       "      <td>180.000100</td>\n",
       "      <td>0.481000</td>\n",
       "      <td>0.50000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>15.853615</td>\n",
       "      <td>8.785562</td>\n",
       "      <td>13414.634022</td>\n",
       "      <td>43.902339</td>\n",
       "      <td>0.499889</td>\n",
       "      <td>0.50025</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>32.600000</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>13996.500000</td>\n",
       "      <td>104.780000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>51.360000</td>\n",
       "      <td>29.000000</td>\n",
       "      <td>47031.802500</td>\n",
       "      <td>138.830000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>68.215000</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>57012.300000</td>\n",
       "      <td>183.130000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.50000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>78.547500</td>\n",
       "      <td>42.000000</td>\n",
       "      <td>65470.635000</td>\n",
       "      <td>218.792500</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>91.430000</td>\n",
       "      <td>61.000000</td>\n",
       "      <td>79484.800000</td>\n",
       "      <td>269.960000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.00000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Daily Time Spent on Site          Age   Area Income  \\\n",
       "count               1000.000000  1000.000000   1000.000000   \n",
       "mean                  65.000200    36.009000  55000.000080   \n",
       "std                   15.853615     8.785562  13414.634022   \n",
       "min                   32.600000    19.000000  13996.500000   \n",
       "25%                   51.360000    29.000000  47031.802500   \n",
       "50%                   68.215000    35.000000  57012.300000   \n",
       "75%                   78.547500    42.000000  65470.635000   \n",
       "max                   91.430000    61.000000  79484.800000   \n",
       "\n",
       "       Daily Internet Usage         Male  Clicked on Ad  \n",
       "count           1000.000000  1000.000000     1000.00000  \n",
       "mean             180.000100     0.481000        0.50000  \n",
       "std               43.902339     0.499889        0.50025  \n",
       "min              104.780000     0.000000        0.00000  \n",
       "25%              138.830000     0.000000        0.00000  \n",
       "50%              183.130000     0.000000        0.50000  \n",
       "75%              218.792500     1.000000        1.00000  \n",
       "max              269.960000     1.000000        1.00000  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ad_data.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Logistic Regression\n",
    "\n",
    "Now it's time to do a train test split, and train our model!\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "** Split the data into training set and testing set using train_test_split**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]\n",
    "y = ad_data['Clicked on Ad']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "** Train and fit a logistic regression model on the training set.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logmodel = LogisticRegression()\n",
    "logmodel.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Predictions and Evaluations\n",
    "** Now predict values for the testing data.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = logmodel.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "** Create a classification report for the model.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.86      0.96      0.91       162\n",
      "           1       0.96      0.85      0.90       168\n",
      "\n",
      "    accuracy                           0.91       330\n",
      "   macro avg       0.91      0.91      0.91       330\n",
      "weighted avg       0.91      0.91      0.91       330\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(logmodel, open('logr.pkl', 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
