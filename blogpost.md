# Pen and paper Bayes

This story assumes you have a basic knowledge of the [Bayesian framework](https://en.wikipedia.org/wiki/Bayesian_inference) - as opposed to the classic frequentist one. But that is pretty much it :)

## Off the beaten bayesian path

The main idea of Bayesian learning is to start with some more or less informed guestimate on how your data looks like, and refine this hypothesis as you see more and more data coming in. Some very basic notation:
- $x_i$ is the $i$-th data point we see, coming from a random variable $X$ we are trying to learn about
- $\bar\theta$ is the vector of the model parameters, so the parameters of the distribution $D$ in $X \sim D(\bar\theta)$ - our goal is to discover reasonable values for it; if $X \sim N(\mu, \sigma^2)$ then $\bar\theta=(\mu, \sigma^2)$
- $\bar\alpha$ is the vector of parameters for the model $H$ ruling $\bar\theta$: as we see $\bar\theta$ as a random vector it follows a distribution whose parameters are represented by $\bar\alpha$, so $\bar\theta \sim H(\bar\alpha)$, the hyperparameters

Note: we will use $\alpha$ in our example below but this is by coincidence.

The machinery relies on the definition of a few items:
1. prior distribution: this is the initial hypothesis $H$, defined as a distribution on the parameters of the data distribution, so $p(\bar\theta|\bar\alpha)$; its parameters are the hyperparameters of the model
2. posterior distribution: this is the update of the initial hypothesis in the light of new data points coming in, so $p(\bar\theta|x_1,x_2,\dots,x_i,\bar\alpha)$; its hyperparameters are a function of the original hyperparameters we initialized the prior distribution with, and the data points of course
3. sampling distribution: this is how your actual data distribution $D$ looks like, and it is known as likelihood; it is $p(x_i|\bar\theta)$
4. marginal likelihood: the sampling distribution, marginalized over the hyperparameters, this is known as evidence, $p(x_i|\bar\alpha)$; we are asking ourselves how a specific instance of the prior is compatible with the data
5. posterior predictive distribution: this is the learning engine, as it defines the probability of an event under any likelihood assumption - not just the current one, which is the latest set of posterior hyperparameters - it is $p(x_i|x_1,x_2,\dots,x_{i-1},\bar\alpha)$; this is main conceptual difference from a frequentist approach based on a simple maximum likelihood.

You can sample some data points from both the current likelihood or the posterior predictive, but there is a fundamental difference: the former accounts only for the specific assumption of the updated parameters as of now, whereas the latter accounts for the fact that a point could still come from _any other_ likelihood - we do not know the truth even if one option emerges as more data points get seen.

Pay attention about the difference between _posterior distribution_ and _posterior predictive distribution_, as you see above they are _not_ synonyms. In particular:
 - the posterior is the repeated update of the prior as new data points come in; it is of the form $p(\bar\theta|x_i)$, it says something on the possible model parameter $\bar\theta$ conditioning on seeing a point $x_i$
 - the posterior predictive says something on the actual point $x_i$, considering all of the possible model hypotheses $\bar\theta$ together (marginalization)

The posterior distribution is one on the parameters of the data distribution, this is why we say that in a Bayesian framework we do not try to predict exactly what the next point will be - coming up with a single best candidate is the frequentist maximum-likelihood based approach - but rather informing how it should look like, so its distribution.

In general updating the posterior is computationally challenging as you basically have to solve integrals. For arbitrary choices of priors and likelihoods you may have to get some help with libraries like [PyMC3](https://github.com/pymc-devs/pymc3). But... for a number of known combinations closed forms for the posterior do exist: this is a big advantage, because it opens the adoption to online learning - when it is just sums and products you can hope to keep up with the data stream, you do not get stranded by expensive numerical stuff.

### A clarifying example

Such magic pairs are called [conjugate priors](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions) and understanding a few examples on either table is an excellent way to make sure you understand what's going on. What is a conjugate prior exactly? The likelihood function (see above) is our hypothesis on the process generating the data: if we are studying a coin flip, the process will be a Bernoulli one - each variable representing a specific coinflip is an indicator, 0 for head 1 for tail. The parameter of a bernoullian is the probability $p$ of heads. The prior distribution sets an initial, hopefully informed, hypothesis on how this $p$ should look like, in the form of a probability distribution. Every time a coin is flipped, so as new data observations come in, this prior is updated, the resulting distribution being called posterior. If the newly updated posterior is from the same family as the prior, then we say that the prior distribution is a _conjugate prior_ for the chose process likelihood distribution. What a family of distributions actually is might be a purely formal and rather vanishing concept. It is a given set of distributions that can be described all together at once with a finite number of parameters. For example the normal distribution may be seen as a _family_ of distributions, specifically, an exponential one; if you can obtain two distributions from the same analytical form by setting their parameters properly, then they are in the same family.

For example, let us say that we think our data points come from a [Pareto](https://en.wikipedia.org/wiki/Pareto_distribution) distribution, so this will be our likelihood. This fat-tail distribution is defined by a scale $x_m$ - telling us what the minimum is - and a shape $k$ - the higher the closer to the axes (note: Wikipedia uses $x_m$ for the scale but $\alpha$ for the shape, for which we use $k$ instead). It is useful to model any event whose samples cannot be negative and for which we expect that large values are not that unlikely (as in: better than what we could do with a [half-normal](https://en.wikipedia.org/wiki/Half-normal_distribution), informally).

Let us fix the scale; in most applications this is a reasonable thing: say we are interested in modeling the expected costs of a claim for an insurer, the minimum can be taken as the processing and office costs of a rejected claim (the insurance does not pay the client but even saying no has some costs). Usually we study a conjugate only after fixing a subset of parameters; in general it is possible to work out a form to learn all of them, and we will briefly touch upon this. Say we fix the scale to some value $x_m=x_m^*$. Please note that for all calculations in our examples we will round to two decimals (this introduces some little rounding errors here and there).

So Pareto is our likelihood, scale is fixed; what is our initial guess for the shape? How should the distribution for the shape value look like, so our prior? From the table we see that the [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) is the one; this distribution has two parameters, $\alpha$ and $\beta$, our hyperparameters for the learning model. A good choice for our Pareto is $k=2.1$, with $2$ as the pathological value where the variance is undefined, so $2+0.1$ to stay sufficiently close. To center the gamma on $2.1$: we know the mean of a gamma is $\mu=\frac{\alpha}{\alpha+\beta}$ so we set:
$\frac{\alpha}{\alpha+\beta}=2.1$ to center around it, and fix $\alpha=2$
(we have no good guess here so we pick a nice $\alpha$) so we get:
$\beta = -1.04$. So the result is $Gamma(2,-1.04)$:
good, we have decided what our prior looks like! In the table you also see how we can update these two hyperparameters as we read in more $x_i$ data points, where $n$ is the number of observations so far. How does our posterior on top the updated $\alpha$ and $\beta$ look like? By the beauty of conjugation, updating a Gamma after seeing data from the Pareto she inspires still yields us a Gamma, with the updated parameters. Notice how the posterior _predictive_ is missing from the table. That means that we can keep generating data from the likelihood process, but that we do not really have a way (yet!) to say something about the probability of a specific data point to come up next, without restricting to a specific hypothesis on the likelihood. 

### Chaining parameters

For likelihoods with multiple parameters as the example above, the default setting is to fix all of them except the one to learn, which becomes the model parameter. This is for one-parameter conjugates. Nothing prevents us from trying to get a sense of all the parameters at once. A quite detailed discussion can be found [here](https://vioshyvo.github.io/Bayesian_inference/chap-multi.html). Take for example a Gaussian conjugate, where both the prior and likelihood are normal. We can have a first guess for the mean:
$\theta \sim N(\mu_0, \tau_0)$
and plug this in for the actual posterior $X \sim N(\theta,\sigma^2_0)$
starting thus with three hyperparameters, two to get the mean (a mean and a variance, as we have a normal prior), one to get the the point after plugging in this mean (a variance, again under a normal prior). See [here](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf) for the worked out example (par. 3). Pay attention: the precision is the inverse of the variance, $\tau=\frac{1}{\sigma^2}$.

## Show me the data

We will now bring the Pareto example to life, showing both the analytical likelihood updates and the numerical predictive posterior, in absence of a safe closed form for the latter.

### Conjugate case

You may want to keep the [example notebook](??) open.

For the Pareto example, on the Wikipedia page we find a very interesting (and unclaimed, at least at the moment of writing) reference: the amount of time a user on Steam will spend playing different games is Pareto distributed. According to the famous principle, this basically means that the average user will spend 80% of the time playing on 20% of the games he or she has - some games get played a lot but most not really. The _basically_ here really has a meaning, since only for a very specific choice of scale and shape we get the classic 80-20 behavior for a Pareto distribution - and we do not know if this is going to be the case for our dataset. We want to study the probability that a game gets played $x$ hours.

It turns out (it is really a coincidence) that on Kaggle we can find the perfect [dataset](https://www.kaggle.com/tamber/steam-video-games?select=steam-200k.csv): the user behavior of hundreds of Steam users.

Peeping at the data we see that the minimum amount of playtime found is 0.1 hours (if you are fed up after five minutes already that was a very bad game indeed); this will become our fixed scale, $x_m=0.1$. In real life you must just guess as you do not have all of the data beforehand, so you cannot take the minimum of course! Or you have to rely on the laws or hard limits of what you are modeling. Chasing the shape parameter $k$ will craft our Pareto. What is our initial guess for the Gamma shaping our $k$? We can start with an educated guess based the average game time we expect in general; from the metadata of this dataset on the website it is unclear to which period it refers to, or if it was ever updated; we can just choose in absolute terms, so from 2003 (Steam launch) to 2017 (dataset creation). Shall we say 7 hours? As a side: is it really a problem if our guess turns out to be inaccurate? If we have enough data and a number of boring conditions are checked, it is not, as in: any bias coming from an initial bad choice in the prior will be washed away as we read more data and adjust the posterior; this is thanks to [this](https://en.wikipedia.org/wiki/Bernstein%E2%80%93von_Mises_theorem) theorem, which tells us that the prior becomes irrelevant, asymptotically. One could even talk about the convergence rate of a Bayesian model, but that is another story. We may not have enough data but the regularity checks are there. So how do we get the desired shape? First we must ask that $k\ge1$, otherwise $\mu=\infty$ which is not true in our case: the average game tends to be played for a finite amount of hours, and a small one too, which we chose to be 7. Plugging in $x_m=0.1$ into the mean for this case we solve for $k$ in the mean for a Pareto which is:

$\begin{cases}
\infty & \text{for }k\le 1 &\\
\dfrac{k x_\mathrm{m}}{k-1} & \text{for }k\gt1
\end{cases}$

and this gives us $k=1.01$. To get started we expect our data process to be a $Pareto(0.1, 1.01)$.

![](initial_pareto.png)


Each data point we take from the dataset is the registered playtime, for some user and some game; as we are interested in the global behavior we discard the user and game information, keeping the hours record only.

Now the question: what are the good values for our prior? Our Pareto parameters are $x_m=0.1$ and $k=1.01$. For the Gamma we set $\alpha=1$ as it is the smallest and nicest shape (for $\alpha\geq1$ the distribution is well defined and has a mode too); observing that our target Pareto shape is $k=1.01$ we will center the Gamma around $k$ by choosing $\beta=0.99$ observing that $\mu=\frac{\alpha}{\beta}$. So again our Pareto scale $x_m$ is fixed and the shape, which is the unknown model parameter, is assumed to behave as a $Gamma(1, 0.99) (here expressed in scale $\theta$, which is the reciprocal of our rate $\beta$)

![](initial_gamma.png)


$. Sampling from this distribution gives a single instance of our likelihood in the form of the Pareto distribution.

And now, time to get on the data. The set is sorted by user id, and it is just a small sample of the users. We are interested in the play events only. As we want to have the feeling of a random sampling from the unknown and Pareto-assumed distribution we will shuffle the rows a little.

Here is the the evolution of the Gamma posterior mean for the first few iterations, so the plot of $\mu=\frac{\alpha}{\beta}$ as the hyperparameters get updated:

![](gamma_mean_100i.png =750x500)


you can see it stabilizes pretty fast around 0.25, realizing the expected value for the shape parameter $k$ of our Pareto likelihood.

Here are the final results for our posterior and likelihood:

![](final_gamma.png)

![](final_pareto.png)

So what we got eventually is a Pareto distribution with scale $x_m=0.1$ and shape $k=0.25$. This is not great after all: if the scale is less than 1 both the mean and variance tend to infinity, and these are two important features to get a feeling for any distribution. If this was an insurance risk model we should really back off as this would mean: the expected claim cost is gigantic!

In our example we do see that a legit Pareto emerges, albeit one which is not very informative and nice to look at. It could be that with more data the result shifts to a nicer model, namely one where the shape is greater than 1, so that the main moments are finite. But as we have converged pretty soon on this dataset, we would really like to see more data from the middle of the distribution, and not from the fat tails - if things really stay too widespread we cannot say anything interesting, an obese tail is a flat line, and we could land anywhere.

### Numerical case

See the example notebook [here](??).

As we have no analytical form for the predictive posterior, using the conjugate approach we can discover how the likelihood should look like, and we can take samples from specific realizations of such best likelihoods too, but... we cannot take a glance at how some anonymous data point would look like, anonymous in the sense of originating from our newly found specific likelihood or any other one like it - so where the shape is still from the updated Gamma.

To get some help here we will turn to numerical sampling. Detailing how this kind of approaches works falls out of scope for our chat. We will use the famous PyMC3 library.

The definition of our model is pretty simple:
```python
prior = pymc3.Gamma('prior', alpha=1, beta=0.99)
likelihood = pymc3.Pareto('likelihood', alpha=prior, m=0.1, observed=play_data)
```

We extract different kinds of samples from our Bayesian building blocks, the last is of course what we were missing:
```python
sampled_posterior_predictive = pymc3.sample_ppc(sampled_posterior)
```

Plotting the prior we see something we knew already: the shape of the Pareto comes to sit around $k=0.25$:

![](pymc3plot.png)

Nice to see that we get to the same results!

A number of samples are offered for the posterior predictive:
```python
sampled_posterior_predictive['likelihood'].shape
```

You may have noticed how the numerical sampling takes more time than the pen & paper recurrence of course.

## Recap

We have had a look at an example of Bayesian learning on real data, taking a chance to see an analytical approach next to the numerical one. Are you happy with the model we have come to? If not, why? :)

See you the $t+1$ time!
