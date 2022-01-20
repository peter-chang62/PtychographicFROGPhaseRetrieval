function [Si] = makePFROG(Et, Gt, Omega, Delay)

Si=zeros(length(Omega),length(Delay));

for i=1:length(Delay)
    
   taui=Delay(i); 
   Gw=fftshift(fft(fftshift(Gt)));
   Gttau=fftshift(ifft(fftshift(Gw.*exp(1i*Omega'*taui))));
   Sit=Gttau.*Et;
   Siw=fftshift(fft(fftshift(Sit)));
   Si(:,i)=abs(Siw).^2;
    
end