r=1;
PT_time=1000;
lambda=0.0001
temp=[];
for i=1000:1000:17000
r=r.*exp(-(i-PT_time).*lambda);
temp=[temp;r];
end