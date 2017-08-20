function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
htheta = sigmoid(X*theta);
J_wo_reg = -1.*y.*log(htheta) - (1-y).*log(1-htheta);
J_wo_reg = sum(J_wo_reg) ./ m;

target_theta = theta(2:size(theta,1),:);
J_reg = sum(target_theta .^ 2) .* lambda ./ (2*m);

J = J_wo_reg + J_reg;

temp = htheta-y;
temp_m = repmat(temp, 1, size(theta));
grad_wo_reg = sum(temp_m .* X) ./ m;

grad_reg = lambda ./ m .* theta';
grad_reg(1,1) = 0;

grad = grad_wo_reg + grad_reg;





% =============================================================

end
