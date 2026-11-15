# MarketPulse: Can News Help Predict the Stock Market?

**Track:** Finance
**Team:** Brijesh Kalani, Krishay Ranawat, Nandan Ranganathan

*This is a school prototype. It is not financial advice and it does not tell anyone to buy or sell.*

## The problem
Our question is: can news help predict the stock market? A lot of beginners read one headline and think they know where a stock will go. Lots of websites also show a single "prediction" number, which makes a guess look like a fact. We think that is risky for people who are just starting to invest. Before we can say if news helps, we first need to know how well a stock can be predicted without news, and how wrong the guess can be.

## What we built
MarketPulse is a website where people with a free account can look at 4,616 stocks. On each stock page you can see the real price history, with little markers where news happened. You also see a predicted estimate for 3, 6 and 12 months. We show it as a middle guess with a likely range instead of one number, plus a High, Medium or Low badge that tells you how much history the estimate is based on. Some big companies also have recent news and a short box about what we think the news means. That box is just our opinion and we have not proven it works. There is also a "Today in 60 seconds" news summary, a portfolio, a watchlist and some beginner lessons. The prices and estimates update every day.

## What we added
We started from the suggested project "News headlines and the stock market." It says to report the answer honestly, even if the answer is "not really." We did the first step: we built a simple model that uses no news at all and tested it. That gives us a score that any news idea would have to beat.

We have not answered the news question yet. This project is the starting point, and testing news is our next step.

Our website, the model and testing in this code were built on 14 and 15 November 2026.

## How it works
For data we used 103,441 monthly closing prices from 400 US stocks. We picked them at random from stocks with at least 9 years of history, and we removed missing months and stocks with too little history. One problem is that we only used stocks that still exist today, which makes the results look a bit better than real life.

For the model, each stock's average monthly move and how much it jumps around come from its own last 10 years. If a stock has less history, we pull its numbers closer to a normal value so we do not trust too little data. The model gives a middle estimate and an 80% range.

## Results
We trained the model on predictions up to December 2019. We tested it on 4,823 predictions from January 2021 to November 2025, which it had never seen.

| Test | Result |
|---|---|
| Always guess "up" | right 50.3% of the time |
| Our model (no news) | right 55.0% of the time |
| Machine learning model (gradient boosting) | right 51.7% of the time |
| Our 80% range held the real price | 78.9% of the time |

So a simple model with no news is only a little better than a coin flip, and any news idea has to beat about 55%. Our model is weak in some places. Its middle estimate is no better than just guessing "no change," so the range is the useful part. The range also worked worse for very jumpy stocks (72.6%). The 55% could partly be luck, and all the data comes from one time period.

## What we would do next
1. Test the real question: do news headlines beat the 55% score? We would use a news dataset and split the data by date.
2. Make the range better for jumpy stocks and test it on stocks that were removed from the market.

## Sources
Price data from Google Finance. Code libraries: pandas, NumPy, scikit-learn. We used Claude (claude.ai) to help write the code and the website, and we ran the code and checked the results ourselves. Idea from the Veritas AI Hackathon 2026 Project Briefs, section 4.2.4.
