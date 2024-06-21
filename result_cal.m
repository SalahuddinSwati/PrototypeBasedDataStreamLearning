function [ EVAL ] = result_cal( Result )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
           EVAL(1,1)= Result.Accuracy;
           EVAL(1,2)= Result.Sensitivity;
           EVAL(1,3)=  Result.Specificity;
           EVAL(1,4)= Result.Precision;
           EVAL(1,5)= Result.Sensitivity;
           EVAL(1,6)= Result.F1_score;
           %EVAL(1,6)= Result.MatthewsCorrelationCoefficient;
           EVAL(1,7)= Result.Kappa;

end

