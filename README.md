Repository for a paper on the effect of traders using LASSO to forecast returns on asset prices.



Good, that goal is actually much easier to get via Stein than via the Taylor circus.

Let me restate the setup cleanly and then go straight for the fixed-point argument.

---

## Setup

* (X \sim \mathcal N(0,\sigma^2)) with (\sigma^2>0).
* For each (b \in \mathbb R), define
  [
  g_b(x) = \log!\Bigl(1 - k, e^{c \tanh!\bigl(\tfrac{b x}{c}\bigr)}\Bigr),
  ]
  with (k\in(0,1)),
  [
  c = -\log k - \varepsilon,\quad \varepsilon>0.
  ]

You want to show that, as a function of (b),
[
f(b) := \mathbb E[X,g_b(X)]
]
satisfies
[
f(b) = b \quad \Longrightarrow \quad b = 0.
]

---

## 1. Stein’s lemma representation

For any fixed (b), (g_b) is differentiable and grows nicely, so Stein’s lemma applies:

[
\mathbb E[X,g_b(X)] = \sigma^2,\mathbb E[g_b'(X)].
]

So we need to understand the sign of (\mathbb E[g_b'(X)]) as a function of (b).

---

## 2. Compute (g_b'(x)) and its sign

Start from
[
g_b(x) = \log!\Bigl(1 - k, e^{c \tanh!\bigl(\tfrac{b x}{c}\bigr)}\Bigr).
]

Differentiate w.r.t. (x):

[
g_b'(x)
= \frac{-(k e^{c\tanh(bx/c)})\cdot c ,\tanh'!\bigl(\tfrac{b x}{c}\bigr)\cdot \tfrac{b}{c}}
{1 - k e^{c\tanh(bx/c)}}.
]

Use (\tanh'(z) = \operatorname{sech}^2(z)):

[
g_b'(x)
= -,\frac{k,b, e^{c\tanh(bx/c)},\operatorname{sech}^2(bx/c)}
{1 - k e^{c\tanh(bx/c)}}.
]

Now check the ingredients:

* (k>0),
* (b) has the same sign as we choose,
* (e^{c\tanh(bx/c)}>0) for all (x),
* (\operatorname{sech}^2(bx/c)>0) for all (x),
* denominator (1 - k e^{c\tanh(bx/c)}) is **strictly positive**:

  * (\tanh(bx/c)\in(-1,1)),
  * so (c\tanh(bx/c)\in(-c,c)),
  * hence (k e^{c\tanh(bx/c)}\in (k e^{-c}, k e^{c})).
  * Now (c=-\log k-\varepsilon\implies k e^{c} = k e^{-\log k-\varepsilon}=e^{-\varepsilon}<1).
    Thus (k e^{c\tanh(bx/c)} \le e^{-\varepsilon}<1), so
    [
    1 - k e^{c\tanh(bx/c)} \ge 1 - e^{-\varepsilon} > 0.
    ]

So the denominator is positive and the whole sign comes from (-b).

Therefore, for every (x\in\mathbb R),

[
\operatorname{sign}\bigl(g_b'(x)\bigr) = -,\operatorname{sign}(b),
]
and in fact (g_b'(x)) is **strictly** negative for all (x) if (b>0), strictly positive for all (x) if (b<0), and identically 0 only when (b=0).

---

## 3. Sign of (f(b) = \mathbb E[X g_b(X)])

By Stein’s lemma,

[
f(b) = \mathbb E[X g_b(X)] = \sigma^2 \mathbb E[g_b'(X)].
]

Now:

* If (b>0), then (g_b'(x)<0) for all (x), hence
  (\mathbb E[g_b'(X)]<0), and so
  [
  f(b) < 0.
  ]
* If (b<0), then (g_b'(x)>0) for all (x), hence
  (\mathbb E[g_b'(X)]>0), and so
  [
  f(b) > 0.
  ]
* If (b=0), then (\tanh(0)=0), so
  [
  g_0(x) = \log(1-k) \quad \text{(constant)},
  ]
  and
  [
  f(0) = \mathbb E[X g_0(X)] = \log(1-k),\mathbb E[X]=0.
  ]

So summarizing the sign:

[
\boxed{
\operatorname{sign}\bigl(f(b)\bigr) = -,\operatorname{sign}(b),\quad
f(0)=0.
}
]

---

## 4. Fixed point equation (f(b) = b)

Now look at
[
f(b) = b.
]

* If (b>0), then (f(b)<0), so (f(b)\neq b).
* If (b<0), then (f(b)>0), so again (f(b)\neq b).
* If (b=0), we have (f(0)=0), so (b=0) **is** a solution.

Therefore the only solution is

[
\boxed{b = 0.}
]

That gives you exactly what you wanted: the fixed-point equation (\mathbb E[X g_b(X)] = b) has the unique solution (b=0).

---

### Where the Taylor logic fits

The Taylor expansion we played with before is consistent with this:

* Near (b=0), you get
  [
  f(b) \approx -\frac{k}{1-k},\sigma^2,b + O(b^3),
  ]
  so the *local* slope is negative, i.e. (f(b)) locally points in the opposite direction of (b).

But the Stein + sign-of-derivative argument above is what upgrades that local picture to the **global** uniqueness of the fixed point. The Taylor series alone wouldn’t be enough to rule out some weird second intersection far from zero.
