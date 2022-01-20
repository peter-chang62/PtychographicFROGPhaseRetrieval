clc
clear


%% Load Data and Waves
Data=load('O:\OFM\abijith\RINMeasurement\171027\decades5.txt');

Freq=Data(:,1);  %Frequency in Hz
VPN=Data(:,2); %Intensity Phase Noise in V/sqrt(Hz)

FVS=0.572; %RMS voltage on photodetector

FPN=VPN/FVS; %normalized intensity noise dB/sqrt(Hz)
FPN=FPN.^2; %dBc/Hz

deltaFreq=Freq(2)-Freq(1);

%% Calculatue Timing Jitter From Phase Noise

NDecades=6;
NPnts=1601;

rFPN=flipud(FPN);

for n=1:NDecades
    
   ndeltaFreq=deltaFreq*10^(NDecades-n);
   i=n-1;
   
   for m=1:NPnts
      
      j=m-1; 
      mWave=rFPN(i*1601+1:i*1601+1+j);
      DSum(n,m)=sum(mWave)*ndeltaFreq;
       
   end
    
end

for n=1:NDecades
    
    i=n-1;
    
    if i==0
        
        RIN(i*1601+1:(i+1)*1601)=DSum(n,:);
        
    else
        
        RIN(i*1601+1:(i+1)*1601)=RIN(i*1601)+DSum(n,:);
        
    end
    
end

RIN=sqrt(RIN);
RIN=fliplr(RIN);

figure(1)
loglog(Freq,RIN,'r','LineWidth',2)
xlabel('Frequency (Hz)')
ylabel('Accumlated Intensity noise')

figure(2)
loglog(Freq,FPN)