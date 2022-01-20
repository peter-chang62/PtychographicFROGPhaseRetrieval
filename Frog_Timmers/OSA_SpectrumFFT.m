clc
clear


%% Load Data and Waves
M=csvread('O:\OFM\Timmers\20170608\W0000.csv');

lambda=M(:,1);
SpecL=M(:,2);

%% Convert from Frequency to Wavelength Space

omega=2*pi*300./lambda;

npnts=2^10;
omega1=linspace(omega(end),omega(1),npnts);
deltaomega=omega1(2)-omega1(1);

SpecW=interp1(omega,SpecL,omega1);
SpecW(isnan(SpecW))=0;

%% Zero-padding

omega1a=0:deltaomega:omega1(1)-deltaomega;
omega1b=omega1(end)+deltaomega:deltaomega:10;

SpecWa=zeros(1,length(omega1a));
SpecWb=zeros(1,length(omega1b));

omega2=[omega1a omega1 omega1b];
SpecW2=[SpecWa SpecW SpecWb];

%% Fourier Transform

EfieldW=sqrt(SpecW2);

EfieldT=fftshift(fft(fftshift(EfieldW)));
ITemp=abs(EfieldT).^2;

tnyq=2*pi/deltaomega;

t=linspace(-tnyq/2,tnyq/2,length(omega2));

%% Temporal selection and fit

[~, Ind1]=min(abs(t+100));
[~, Ind2]=min(abs(t-100));

t1=t(Ind1:Ind2);
ITemp1=ITemp(Ind1:Ind2);
ITemp1=ITemp1/max(ITemp1);

gaussEqn = 'A*exp(-4*log(2)*(x-b)^2/c^2)';

GaussianFit=fit(t1',ITemp1',gaussEqn,'StartPoint',[1, 0, 20],'MaxIter',10000,'TolX',1e-6);

GaussFitValues=coeffvalues(GaussianFit);
Width=GaussFitValues(1,3);

figure
plot(GaussianFit)
title(strcat('FTL Width = ',num2str(Width),' fs'));
hold on
plot(t1,ITemp1)
axis([-100 100 -0.01 1.1])
hold off

figure
plot(lambda,SpecL)

