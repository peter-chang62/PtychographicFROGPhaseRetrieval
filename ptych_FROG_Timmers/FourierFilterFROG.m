S=load('O:\OFM\1_MIRLab\UI_ChirpedMirrors\20180612\FROG_1500mA_m200fs_200fs_1fsstep.txt');

AutoCorr=sum(S);
AutoCorr=AutoCorr-AutoCorr(1);
FAutoCorr=fftshift(fft(fftshift(AutoCorr)));

FilterWidth=length(AutoCorr)*20;
FilterCenter=length(AutoCorr)/2;
N=1:length(AutoCorr);
Filter=exp(-(2*sqrt(log(2)))^4*(N-FilterCenter).^4/FilterWidth.^2);

subplot(3,1,1)
plot(abs(FAutoCorr)/max(abs(FAutoCorr)),'k')
hold on
plot(Filter,'r')
hold off

for n=1:size(S,1)
    
   LineOut=S(n,:);
   FLineOut=fftshift(fft(fftshift(LineOut)));
   FLineOut=Filter.*FLineOut;
   LineOut=fftshift(ifft(fftshift(FLineOut)));
   LineOut=abs(LineOut);
   SFilter(n,:)=LineOut;
    
end

subplot(3,1,2)
surf(S)
view(2)
shading flat

subplot(3,1,3)
surf(SFilter)
view(2)
shading flat

dlmwrite('O:\OFM\1_MIRLab\UI_ChirpedMirrors\20180612\FROG_1500mA_m200fs_200fs_1fsstep_filtered.txt',SFilter,'delimiter','\t')