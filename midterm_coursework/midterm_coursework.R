# ---------------------------------------------------------------
# Midterm coursework script for MATH48011/69011  Linear Models with Nonparametric Regression
# ---------------------------------------------------------------

# Load required libraries
library(tidyverse)  
library(broom)      

theme_lm <- theme_classic() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
    axis.ticks = element_line(color = "black"),
    axis.line = element_line(color = "black")
  )

# Create output directory for figures
output_dir <- "midterm_figures"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# ===============================================================
# Q1. Load the data into R. Draw exploratory plots of the relationship between x, z, and Y.
# Comment on any interesting features in your findings.
# ===============================================================

# ---- Data loading and preparation ----
data <- read_csv("Semaglutide.csv")
data$sex <- as.factor(data$sex) 

# Q1(a) Figure 1.1: Overall relationship between age (x) and MWDR (Y)
plot_q1_1 <- ggplot(data, aes(x = age, y = MWDR)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", formula = y ~ x + I(x^2)) +
  labs(
    title = "Figure 1.1: MWD Ratio vs Age (Overall Trend)",
    x = "Age (x)",
    y = "MWDR(Y)"
  ) +
  theme_lm
print(plot_q1_1)
ggsave(
  filename = file.path(output_dir, "figure_q1_1_overall_trend.png"),
  plot = plot_q1_1,
  width = 6.5,
  height = 5,
  dpi = 300
)

# Q1(b) Figure 1.2: MWDR distribution by sex (categorical comparison)
plot_q1_2 <- ggplot(data, aes(x = sex, y = MWDR, fill = sex)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    title = "Figure 1.2: MWD Ratio vs Sex",
    x = "Sex (z)",
    y = "MWDR (Y)"
  ) +
  guides(fill = "none") +
  theme_lm
print(plot_q1_2)
ggsave(
  filename = file.path(output_dir, "figure_q1_2_boxplot_by_sex.png"),
  plot = plot_q1_2,
  width = 6.5,
  height = 5,
  dpi = 300
)

# Q1(c) Figure 1.3: ANCOVA-style view with quadratic fits by sex
plot_q1_3 <- ggplot(data, aes(x = age, y = MWDR, color = sex)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", formula = y ~ x + I(x^2), se = FALSE) +
  labs(
    title = "Figure 1.3: MWD Ratio vs Age by Sex",
    x = "Age (x)",
    y = "MWDR (Y)",
    color = "Sex"
  ) +
  theme_lm
print(plot_q1_3)
ggsave(
  filename = file.path(output_dir, "figure_q1_3_ancova_view.png"),
  plot = plot_q1_3,
  width = 6.5,
  height = 5,
  dpi = 300
)

# ===============================================================
# Q2. Create a dummy variable w using male as the reference level. Fit model F
# ===============================================================

# Q2 Step 1: Dummy variable w (female = 1, male reference = 0)
data_model <- data
data_model$w <- ifelse(data_model$sex == "F", 1, 0)

# Q2 Step 2: Fit Model F -> E[Y] = θ0 + θ1 x + θ2 x^2 + θ3 w + θ4 x w
fit_F <- lm(MWDR ~ age + I(age^2) + w + age:w, data = data_model)

# Q2 Step 3: Tidy coefficients for reporting
tidy_fit <- tidy(fit_F)
print(tidy_fit)

# Q2 Figure 2.1: Observed data with fitted curves by sex
plot_data_Q2 <- augment(fit_F, data = data_model)
plot_data_Q2 <- plot_data_Q2[order(plot_data_Q2$sex, plot_data_Q2$age), ]

plot_q2_1 <- ggplot(plot_data_Q2, aes(x = age, y = MWDR, color = sex)) +
  geom_point(alpha = 0.5) +
  geom_line(aes(y = .fitted, group = sex), linewidth = 1) +
  labs(
    title = "Figure 2.1: Model F Fitted Curves",
    x = "Age (x)",
    y = "MWDR (Y)",
    color = "Sex"
  ) +
  theme_lm
print(plot_q2_1)
ggsave(
  filename = file.path(output_dir, "figure_q2_1_modelF_fit.png"),
  plot = plot_q2_1,
  width = 6.5,
  height = 5,
  dpi = 300
)

# ===============================================================
# Q3. Give the interpretation of the parameters θ0, θ0 + θ3, θ4. 
# ===============================================================

# Q3: Extract needed coefficients
theta_hat <- coef(fit_F)
theta0_hat <- theta_hat["(Intercept)"]
theta1_hat <- theta_hat["age"]
theta2_hat <- theta_hat["I(age^2)"]
theta3_hat <- theta_hat["w"]
theta4_hat <- theta_hat["age:w"]

# Q3 Table: tidy summary for the write-up
coefficinet_table <- tibble(
  quantity = c(
    "θ0 (x=0, Male baseline)",
    "θ0 + θ3 (x=0, Female)",
    "θ4 (interaction effect)"
  ),
  estimate = c(
    theta0_hat,
    theta0_hat + theta3_hat,
    theta4_hat
  )
)
print(coefficinet_table)

# ===============================================================
# Q4. Model assumptions and diagnostics 
# ===============================================================

# Q4 setup: collect fitted values and residuals once
diag_data <- augment(fit_F)

# Q4 Figure 4.1: Residuals vs fitted
plot_q4_1 <- ggplot(diag_data, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Figure 4.1: Residuals vs Fitted Values",
    x = "Fitted values (.fitted)",
    y = "Residuals (.resid)"
  ) +
  theme_lm
print(plot_q4_1)
ggsave(
  filename = file.path(output_dir, "figure_q4_1_residuals_vs_fitted.png"),
  plot = plot_q4_1,
  width = 6.5,
  height = 5,
  dpi = 300
)

# Q4 Figure 4.2: Normal Q-Q plot
plot_q4_2 <- ggplot(diag_data, aes(sample = .resid)) +
  geom_qq() +
  geom_qq_line(color = "red") +
  labs(
    title = "Figure 4.2: Normal Q-Q Plot of Residuals",
    x = "Theoretical quantiles",
    y = "Sample quantiles"
  ) +
  theme_lm
print(plot_q4_2)
ggsave(
  filename = file.path(output_dir, "figure_q4_2_residuals_qq.png"),
  plot = plot_q4_2,
  width = 6.5,
  height = 5,
  dpi = 300
)

# ===============================================================
# Q5. Hypothesis test
# ===============================================================

# Q5 coefficients entering the linear combination λ = θ3 + 40θ4
theta_hat_3 <- coef(fit_F)["w"]
theta_hat_4 <- coef(fit_F)["age:w"]

estimate <- theta_hat_3 + 40 * theta_hat_4  

V <- vcov(fit_F)  
var_lambda_theta <- V["w", "w"] +
  (40^2) * V["age:w", "age:w"] +
  2 * 40 * V["w", "age:w"]
se_lambda_theta <- sqrt(var_lambda_theta)

t_stat <- estimate / se_lambda_theta
df_residual <- df.residual(fit_F)  
p_value <- 2 * pt(abs(t_stat), df = df_residual, lower.tail = FALSE)

hypothesis_result <- tibble(
  contrast = "θ3 + 40θ4",
  estimate = estimate,
  se = se_lambda_theta,
  t_statistic = t_stat,
  df = df_residual,
  p_value = p_value
)
print(hypothesis_result)

# ===============================================================
# Q6. Prediction for age = 22, male (w = 0) with 95% PI 
# ===============================================================

# Q6 new subject description (age 22, male => w = 0)
new_data_point <- tibble(age = 22, w = 0)

# Q6 compute 95% prediction interval from Model F
prediction_result <- predict(
  fit_F,
  newdata = new_data_point,
  interval = "prediction",
  level = 0.95
)
print(prediction_result)

# Q6 Figure 6.1
plot_data_base <- augment(fit_F, data = data_model)
plot_data_base <- plot_data_base[order(plot_data_base$sex, plot_data_base$age), ]

plot_data_Q6 <- as_tibble(prediction_result)
plot_data_Q6$age <- 22
plot_data_Q6$w <- 0
plot_data_Q6$sex <- factor("M", levels = levels(data_model$sex))

plot_q6_1 <- ggplot(plot_data_base, aes(x = age, color = sex)) +
  geom_point(aes(y = MWDR), alpha = 0.4) +
  geom_line(aes(y = .fitted, group = sex), linewidth = 1) +
  geom_point(
    data = plot_data_Q6,
    aes(x = age, y = fit),
    color = "black",
    fill = "red",
    size = 4,
    shape = 23
  ) +
  geom_errorbar(
    data = plot_data_Q6,
    aes(x = age, ymin = lwr, ymax = upr),
    color = "black",
    width = 0.8,
    linewidth = 1
  ) +
  labs(
    title = "Figure 6.1: Model Fit with 95% Prediction Interval (Age 22, Male)",
    x = "Age (x)",
    y = "MWDR (Y)",
    color = "Sex"
  ) +
  theme_lm
print(plot_q6_1)
ggsave(
  filename = file.path(output_dir, "figure_q6_1_prediction_interval.png"),
  plot = plot_q6_1,
  width = 6.5,
  height = 5,
  dpi = 300
)

pdf(file.path(output_dir, "all_figures.pdf"), width = 8, height = 6)
print(plot_q1_1)
print(plot_q1_2)
print(plot_q1_3)
print(plot_q2_1)
print(plot_q4_1)
print(plot_q4_2)
print(plot_q6_1)
dev.off()