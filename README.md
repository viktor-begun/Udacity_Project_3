# Analyze A/B Test Results

**Purpose:** to understand the results of an A/B test run by an e-commerce website. The company has developed a new web page in order to try and increase the number of users who "convert," meaning the number of users who decide to pay for the company's product. The goal is to help the company understand if they should implement this new page, keep the old page, or perhaps run the experiment longer to make their decision.

For more information on A/B Tests see, for example, [the wiki page](https://en.wikipedia.org/wiki/A/B_testing) and the [figure below](https://en.wikipedia.org/wiki/A/B_testing#/media/File:A-B_testing_example.png). 
<img src="https://upload.wikimedia.org/wikipedia/commons/2/2e/A-B_testing_example.png" width="100%">

### Conclusions:
#### Part I - Probability 
- The probability to convert is approximately 12% for the control and for the treatment group. The size of the group is large - about 30000 individuals. There is NO sufficient evidence to conclude that the new treatment page leads to more conversions. **No need to introduce the new page.**

#### Part II - A/B Test

- The p-value is larger than 5%. There is no evidence to reject the null hypothesis at the assumed Type I error rate of 5%.

- The z-score tells us that the convert_old and convert_new have the difference lower than 1.32 standard deviations. The p-value is larger than the assumed alpha value. Therefore, there is no evidence to reject the null hypothesis. These findings agree with the previous findings.

It means that **the second method gave the same result. There is NO need to introduce the new page.**

#### Part III - A regression approach

The p-value associated with ab_page is 0.19. It is close to the value given by the "ztest", and approximately twice as large as the value found in Part II. This might the cause of the one-sided p-value calculated in Part II, and two-sided p-value calculate in the z-test and by the "sm.Logit()".

The data can be described by only one parameter - intercept. The p-value for other values is much larger than the assumed alpha value of 0.05. **There is NO evidence that country had an impact on conversion.**

**The data suggest that the company may increase the revenue only by increasing the number of users.**

**Prerequisites:** [Python](https://www.udacity.com/) enviroment (e.g. [Anaconda](https://www.anaconda.com)), [Jupyter Notebook](https://jupyter.org/), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Random](https://docs.python.org/3/library/random.html), [Matplotlib](https://matplotlib.org), see the `requirements.txt` file for the list of all installed packages.
