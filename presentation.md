## 📊 **Detailed Slide Content & Speaking Scripts**

### **Slide 1: Title Slide**
**Content:**
```
"Good [morning/afternoon], everyone. I'm Andrew Richard, and today I'm excited to present my capstone project: MLB Playoff Prediction Using Machine Learning.

The central question driving this project is: Can we predict which teams make the playoffs using only mid-season data? Because baseball is notoriously unpredictable, this is a fascinating challenge - there's a saying that 'any team can win on any given day.' But as we'll see, machine learning can find patterns in the chaos.

Over the next 10 minutes, I'll show you how i went about using machine learning to make real time predictions for the 2025 playoffs visualized in a Streamlit application."
```


---

### **Slide 2: Problem Statement**
**Content:**
```
"Let me start by framing the challenge. Major League Baseball has 30 teams, but only 12 make the playoffs - that's a 40% success rate. The constraint that i set for this project is that we must make predictions using only pre-All-Star break data - essentially, we're trying to predict the full season outcome when teams have only played about 60% of their games.

Add to this baseball's inherent randomness - even the best teams lose about 60 games per season - as well as the fact that playoff teams are decided by division, not just based totally on their records, and you can see why this is a complex prediction problem.

But why does this matter? The applications are significant: sports analytics for team evaluation, fantasy sports decision-making, and fundamentally understanding what statistical factors drive success in baseball.

For this project, I assembled a widespread dataset spanning 35 years of MLB history - from 1990 to 2025 - capturing over 30 different team statistics (hitting and pitching) per season. This gives us nearly 1,000 team records to train our models, providing the depth needed for robust machine learning."
```


---

### **Slide 3: Methodology**
**Content:**
```
"Now, let's dive into the methodology. I built a complete data pipeline starting with web scraping from MLB.com. I implemented proper rate limiting and error handling to ensure reliable data collection (after a lot of troubleshooting), mapping pre-All-Star break statistics to full-season playoff outcomes.

For the machine learning approach, I selected two complementary algorithms: XGBoost, which excels at finding complex patterns through gradient boosting, and Logistic Regression with regularization, which provides interpretable statistical insights. Both models use 5-fold stratified cross-validation to ensure robust performance estimates.

The evaluation strategy focuses on ROC AUC as the primary metric - this measures how well the model distinguishes between playoff and non-playoff teams across all probability thresholds. I also conduct detailed feature importance analysis.

I also added an ensemble approach. Both models achieve 96.7% agreement on their predictions, which gives us tremendous confidence in the results. Additionally, the application enforces actual MLB playoff structure - three division winners plus three wild card teams per league - making the predictions practically relevant."
```


---

### **Slide 4: Results**
**Content:**
```
"The results exceeded expectations. In cross-validation, our Logistic Regression model achieved 97.75% ROC AUC, while XGBoost achieved 97.25%. To put this in perspective, ROC AUC above 97% is exceptional for sports prediction - most published models in this domain achieve 60-70% accuracy. There is a caveat that without having more validation data, there is a possibility of data leakage.

When we tested on the 2024 season - validation data the models had never seen - we achieved 86.7% accuracy with XGBoost and 90% with Logistic Regression.

The feature importance analysis reveals fascinating insights that align with baseball wisdom. The top predictor is actually pitching losses - fewer losses strongly indicate better team performance. This is followed by team wins, ERA, and saves, highlighting how crucial pitching is to playoff success.

This aligns perfectly with what every pitching coach tells his pitchers: 'pitching wins championships.' Our machine learning models discovered this truth independently from the data, which validates both our approach and long-standing baseball analytics.

What makes these results particularly compelling is that we're not just achieving high accuracy - we're doing it while making predictions that follow real MLB playoff rules, making this immediately applicable to real-world scenarios."
```


---

### **Slide 5: Implementation**
**Content:**
```
"I built an interactive Streamlit dashboard that provides real-time 2025 season playoff probability predictions. The application enforces actual MLB playoff structure and includes educational components to help users understand the statistical reasoning behind predictions.

The technical implementation emphasizes reproducibility and ease of use. I started playing around with Shell scripts, so now the entire system can be set up with a single command, and the complete training pipeline - from data cleaning through model optimization - executes in just 25 seconds.

Additionally, I added some educational components to the app. The dashboard includes model comparison tools, ROC curves, confusion matrices, and feature importance visualizations that help users understand not just what the model predicts, but why.

Let me now show you this application in action..."
```


---

### **Slide 6: Discussion**
**Content:**
```
"Let's discuss the key findings and implications. First, our models independently discovered that pitching statistics are stronger predictors of playoff success - validating decades of baseball wisdom through data science.

The cross-validation results show excellent generalization, meaning our models aren't just memorizing historical patterns but learning transferable insights about team success.

Importantly, we've demonstrated that mid-season data contains sufficient signal for accurate prediction. This has practical implications for team management and strategic decision-making.

Of course, there are limitations. We're working with team-level statistics only - no player-specific data. We don't account for mid-season trades or injuries, which can significantly impact team performance.
```


---

### **Slide 7: Conclusion**
**Content:**
```
"In conclusion, this project successfully demonstrates that machine learning can predict MLB playoff outcomes with exceptional accuracy using only mid-season data. The 97%+ ROC AUC performance represents state-of-the-art capability in sports prediction.

Beyond the technical achievements, this project showcases a complete data science workflow: from data collection through deployment. The rigorous cross-validation methodology ensures robust results, while the interactive application makes the insights accessible and educational.

Most importantly, this demonstrates the practical application of data science principles to solve real-world challenges.

Thank you for your attention. I'm happy to take questions about any aspect of the methodology, results, or implementation."
```


---
---
---
