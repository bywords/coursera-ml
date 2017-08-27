function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

bias_term = theta(1);
real_theta = theta(2:size(theta,1));

h = X * theta;
h_minus_y = h - y;
J_wo_reg = 1 / (2*m) * sum((h_minus_y).^2, 1);
J_reg = lambda / (2*m) * sum(real_theta .^ 2, 1);

J = J_wo_reg+J_reg;


grad_theta_zero = sum(repmat(h_minus_y, 1, size(X, 2)) .* X, 1) / m;
grad_add_term = [0 (lambda / m .* real_theta)'];
grad = grad_theta_zero + grad_add_term;










% =========================================================================

grad = grad(:);

end
