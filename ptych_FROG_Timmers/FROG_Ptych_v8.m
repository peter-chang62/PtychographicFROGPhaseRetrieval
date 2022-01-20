%Ptychographic FROG Reconstruction Code, written by Henry Timmers
%Algorithm based on the following papers:
%T. Witting et al, "Time-domain ptychography of over-octave-spanning laser
%pulses in the single-cycle regime," Opt. Lett. 41 (18), 4218-4221 (2016).
%M. Lucchini et al, "Ptychographic reconstruction of attosecond pulses,"
%Opt. Ex. 23 (23), 29502-29513 (2015).
%P. Sidorenko et al, "Ptychographic reconstruction algorithm for
%frequency-resolved optical gating: super-resolution and supreme
%robustness," Optica 3 (12), 1320-1330 (2016).

clc
close all
clear

%%  Loading & Preprocessing of FROG Spectrogram

Iter=500; %Number of FROG iterations
Update=50; %Number of iterations between figure update

S=load('NFROG.txt');

Delay0=-250:2:250;
Wavelength=load('Wavelength2.txt');
Omega=2*pi*300./Wavelength;


%Choose portion symmetric around peak intensity
autocorrelation=sum(S,1);
[~,PeakInd1]=max(autocorrelation);
PeakInd2=length(Delay0)-PeakInd1;
PeakIndVec=[PeakInd1 PeakInd2];
[PeakInd, Which]=min(PeakIndVec);

if Which==1
    
    Delay=Delay0(1:2*PeakInd+1);
    S=S(:,1:2*PeakInd+1);
    
else 
    
    if length(PeakInd)>1
        
    else
    
        Delay=Delay0(length(Delay0)-2*(PeakInd-1):end);
        S=S(:,length(Delay0)-2*(PeakInd-1):end);
        
    end
    
end

%Choose frequency range
% surf(Delay,Omega,S)
% view(2)
% shading flat
% colormap(jet)
% axis([Delay(1) Delay(end) 1.6 3.6])
% 
% w1=input('Begin Frequency Range (rad/fs):   ');
% w2=input('End Frequency Range (rad/fs):   ');

w1=1.9;
w2=3.5;
[~, Ind3]=min(abs(Omega-w1));
[~, Ind4]=min(abs(Omega-w2));
Omega=Omega(Ind4:Ind3);
S=S(Ind4:Ind3,:);
S=S-mean(S(1,:));

%Add zero pad to increase time resolution


%Redimension FROG trace
NWpnts=2^11;
NDpnts=floor(length(Delay)/2)*2;

OmegaExp=linspace(Omega(end),Omega(1),NWpnts);
% OmegaExp=linspace(0.5,5,NWpnts);


for i=1:length(Delay)
    
    S1(:,i)=interp1(Omega,S(:,i),OmegaExp);
    S1(isnan(S1))=0;
    
end

DelayExp=linspace(Delay(1),Delay(end),NDpnts);

for i=1:length(OmegaExp)
    
    S2(i,:)=interp1(Delay,S1(i,:),DelayExp);
    S2(isnan(S2))=0;
    
end

DelayRange=DelayExp(end)-DelayExp(1);
DelayExp=linspace(-DelayRange/2,DelayRange/2,NDpnts);

%Divide Out BBO Phase-matching Curve

PMCurve=load('BBO_50um_PhaseMatchingCurve.txt');
PMLambda=PMCurve(:,1);
PMOmega=2*pi*0.3./PMLambda;
PMEff=PMCurve(:,2);

PMEffExp=interp1(PMOmega,PMEff,OmegaExp);
PMEffExp=PMEffExp';
PMEffExp(isnan(PMEffExp))=1;

for i=1:length(DelayExp)
    
   S2(:,i)=S2(:,i)./PMEffExp; 
    
end

% % Alternative Background Subtract
% 
% Bckgnd=sum(S2(:,1:5),2)/5;
% 
% for i=1:NDpnts
%     
%     LineOut=S2(:,i)-Bckgnd;
%     S2(:,i)=LineOut;
%     
% end

%Symmetrize FROG
symm=input('Do you want to symmetrize the FROG trace (1=yes, 0=no):   ');

if symm==1
    
    SA=S2(:,1:length(DelayExp)/2);
    SB=S2(:,length(DelayExp)/2+1:end);
    SB=fliplr(SB);
    SC=1/2*(SA+SB);
    S2=[SC fliplr(SC)];

end

% % Alternative Background Subtract
% 
% Bckgnd=sum(S2(:,1:5),2)/5;
% 
% for i=1:NDpnts
%     
%     LineOut=S2(:,i)-Bckgnd;
%     S2(:,i)=LineOut;
%     
% end

%Background Subtract & normalization (uses highest frequency line as background)
Bckgnd=S2(end,:);
Bckgnd=sum(Bckgnd)/NDpnts;
S2=S2-Bckgnd;
S2(S2<0)=0;


SExp=S2/max(max(S2));

%% Initial Guess for Pulse and Gate Field

DeltaOmega=OmegaExp(2)-OmegaExp(1);
tnyq=2*pi/DeltaOmega;
t=linspace(-tnyq/2,tnyq/2,NWpnts);
t=t';

Et=sum(SExp,1);
Et=Et';
Et=Et.*exp(1i*rand(NDpnts,1)*pi/8);
Et=interp1(DelayExp,Et,t);
Et(isnan(Et))=0;
Gt=Et;

%% Reconstruction

Pti=Et;
Gti=Gt;

for n=1:Iter/Update
    
    tic
    
    for m=1:Update 
        
      j=randperm(length(DelayExp));
      
      for i=1:NDpnts
          
          jiter=j(i);
          Delayi=DelayExp(jiter);
          
                      % UPDATE PULSE Field
            % time delayed gate pulse
            Gwi = fftshift(fft(fftshift(Gti)));
            Gtishift = fftshift(ifft(fftshift(Gwi.*exp(1i*OmegaExp'*Delayi))));

            % Fourier transform of product field
            chiti = Gtishift.*Pti;
            chiwi = fftshift(fft(fftshift(chiti)));
            chiwi = Denoise(real(chiwi),1e-3)+1i*Denoise(imag(chiwi),1e-3);
        
            % replace modulus with measurement
            chiwiprime = sqrt(SExp(:,jiter)).*exp(1i*angle(chiwi));
        
            % inverse Fourier transformation
            chitiprime = fftshift(ifft(fftshift(chiwiprime)));
        
            % difference
            Deltachiti = chitiprime-chiti;
        
            % update object function
            Pupdate = conj(Gtishift)./max(abs(Gtishift).^2);
            beta_P = randi([10, 30],1);
            beta_P = beta_P/100;
            Ptirecon = Pti+beta_P*Pupdate.*Deltachiti;
            
            %Subtract out linear phase ramp to keep pulse at t=0
%             Ptiw=fftshift(fft(fftshift(Ptirecon)));
%             PtiwAmp=abs(Ptiw);
%             PtiwAngle=unwrap(angle(Ptiw));
%             PtiwAngleLin=(PtiwAngle(end)-PtiwAngle(1))/(OmegaExp(end)-OmegaExp(1))*(OmegaExp-OmegaExp(1))+PtiwAngle(1);
%             PtiwAngle=PtiwAngle-PtiwAngleLin';
%             Ptiw=PtiwAmp.*exp(1i*PtiwAngle);
%             Ptirecon=fftshift(ifft(fftshift(Ptiw)));
            
            [~,Indt0]=max(Ptirecon);
            Ptirecon=circshift(Ptirecon,NWpnts/2-Indt0);
%             t0=t(Indt0);
%             Ptirecon=Ptirecon.*exp(-(2*sqrt(log(2)))^4*(t-t0).^4/(10000^4));
            
            %Update guess for next iteration
            Pti=Ptirecon;
            Pti(isnan(Pti))=0;
            Gti=Ptirecon;
            Gti(isnan(Gti))=0;
      end
      
      
      %Update reconstructed FROG trace and calculate error
      Si=makePFROG(Pti,Gti,OmegaExp,DelayExp);
      Si=Si/max(max(Si));
%       err=sqrt(sum(sum(abs(Si-SExp).^2))/(NDpnts*NWpnts));
      fun = @(gamma) sqrt(sum(sum(abs(Si-gamma*SExp).^2))/(NDpnts*NWpnts));
      [gamma_min,err] = fminsearch(fun,1); % finding error minimum
      
      Err((n-1)*Update+m,1)=err;
      Ptvec(:,(n-1)*Update+m)=Pti;
      Gtvec(:,(n-1)*Update+m)=Gti;    

      disp(m)
      
    end
    
    clc
    
    subplot(3,2,1)
    surf(DelayExp,OmegaExp,SExp)
    view(2)
    axis([DelayExp(1) DelayExp(end) OmegaExp(1) OmegaExp(end)])
    shading flat
    colormap(jet)
    xlabel('Delay (fs)','Fontsize',16)
    ylabel('Frequency (rad/fs)','Fontsize',16)
    set(gcf,'Color',[1 1 1])
    set(gca,'Fontsize',14)
    
    subplot(3,2,2)
    surf(DelayExp,OmegaExp,Si)
    view(2)
    shading flat
    axis([DelayExp(1) DelayExp(end) OmegaExp(1) OmegaExp(end)])
    colormap(jet)
    title(strcat('FROG Error = ',num2str(min(Err))))
    xlabel('Delay (fs)','Fontsize',16)
    ylabel('Frequency (rad/fs)','Fontsize',16)
    set(gcf,'Color',[1 1 1])
    set(gca,'Fontsize',14)
    
    subplot(3,2,3)
    plotyy(t,abs(Pti).^2,t,unwrap(angle(Pti)))
    xlim([-500 500])
    xlabel('Time (fs)','Fontsize',16)
    ylabel('Amplitude (arb. units)','Fontsize',16)
    set(gcf,'Color',[1 1 1])
    set(gca,'Fontsize',14)
    
    subplot(3,2,4)
    plotyy(OmegaExp,abs(fftshift(fft(fftshift(Pti)))).^2,OmegaExp,unwrap(angle(fftshift(fft(fftshift(Pti))))))
    xlim([OmegaExp(1) OmegaExp(end)])
    xlabel('Frequency (rad/fs)','Fontsize',16)
    ylabel('Amplitude (arb. units)','Fontsize',16)
    set(gcf,'Color',[1 1 1])
    set(gca,'Fontsize',14)
    
    subplot(3,2,5)
    plot(Err)
    xlabel('Iteration','Fontsize',16)
    ylabel('Error','Fontsize',16)
    set(gcf,'Color',[1 1 1])
    set(gca,'Fontsize',14)
    
    getframe
    
    toc 
    
end

% close all

%% Post-Processing of Reconstructed Spectrogram

Err(1:10)=1;

%Find Pulse with minimum FROG error
[MinErr, MinErrInd]=min(Err);
Pt=Ptvec(:,MinErrInd);
Gt=Gtvec(:,MinErrInd);

%Filter to ps window

% [~,Indt0]=max(Pt);
% t0=t(Indt0);
% Pt=Pt.*exp(-(2*sqrt(log(2)))^4*(t-t0).^4/(500^4));

Srecon=makePFROG(Pt,Gt,OmegaExp,DelayExp);
Srecon=Srecon/max(max(Srecon));

%Plot experimental & reconstrusted FROG trace
figure(1)
subplot(1,2,1)
surf(DelayExp,OmegaExp,SExp)
view(2)
axis([DelayExp(1) DelayExp(end) OmegaExp(1) OmegaExp(end)])
shading flat
colormap(jet)
title('Experimental FROG Reconstruction')
xlabel('Delay (fs)','Fontsize',16)
ylabel('Frequency (rad/fs)','Fontsize',16)
set(gcf,'Color',[1 1 1])
set(gca,'Fontsize',14)

subplot(1,2,2)
surf(DelayExp,OmegaExp,Srecon)
view(2)
shading flat
axis([DelayExp(1) DelayExp(end) OmegaExp(1) OmegaExp(end)])
colormap(jet)
title(strcat('FROG Error = ',num2str(MinErr)))
xlabel('Delay (fs)','Fontsize',16)
ylabel('Frequency (rad/fs)','Fontsize',16)
set(gcf,'Color',[1 1 1])
set(gca,'Fontsize',14)

%Post-processing of reconstructed pulse
ITemp=abs(Pt).^2;
ITemp=ITemp/max(ITemp);
[~, t0Ind]=max(ITemp);
t0=t(t0Ind);

gaussEqn = 'A*exp(-4*log(2)*(x-b)^2/c^2)';
GaussianFit=fit(t,ITemp,gaussEqn,'StartPoint',[1, t0, 50],'MaxIter',10000,'TolX',1e-6);
XUVFitValues=coeffvalues(GaussianFit);
XUVWidth=XUVFitValues(1,3);

figure(2)
subplot(1,2,1)
plot(t,ITemp)
axis([-1000 1000 -0.025 1.025])
title(strcat('Pulse Duration = ',num2str(XUVWidth),' fs'))
xlabel('Time (fs)')
ylabel('Temporal Envelope')

Ewfield=(fftshift(fft(fftshift(Pt))));
EwAmp=abs(Ewfield).^2;
EwPhase=unwrap(angle(Ewfield));
% EwPhaseLin=(EwPhase(end)-EwPhase(1))/(OmegaExp(end)-OmegaExp(1))*(OmegaExp-OmegaExp(end))+EwPhase(end);
% EwPhase=EwPhase-EwPhaseLin';
omega0=sum(EwAmp.*OmegaExp')/sum(EwAmp);
% [~,IndPeak]=max(EwAmp);
% omega0=OmegaExp(IndPeak);
[~, IndPeak]=min(abs(OmegaExp-omega0));
OmegaExp=OmegaExp-omega0/2;

%Force peak to 1.55 um

ForcePeak=input('Force Peak to 1.550 um? Yes=1, No=0:   ');

if ForcePeak==1
    
    w1=2*pi*0.3/1.55;
    [~,IndFP]=max(EwAmp);
    w0=OmegaExp(IndFP);
    OmegaExp=OmegaExp+w1-w0;
    
end

LambdaExp=2*pi*0.3./OmegaExp;

figure(2)
subplot(1,2,2)
yyaxis left
plot(LambdaExp,(EwAmp))
xlim([1 2])
yyaxis right
plot(LambdaExp,EwPhase)
xlim([1 2])
xlabel('Wavelength (um)')
ylabel('Spectrum')

ReconstructedW=zeros(length(Ewfield),3);
ReconstructedW(:,1)=abs(EwAmp).^2;
ReconstructedW(:,2)=EwPhase;
ReconstructedW(:,3)=LambdaExp';

ReconstructedT=zeros(length(ITemp),3);
ReconstructedT(:,1)=sqrt(ITemp);
ReconstructedT(:,2)=unwrap(angle(Pt));
ReconstructedT(:,3)=t;

%% Saving Data
 
SaveData=input('Would you like to save the Reconstructed FROG Data? Yes=1, No=0:   ');

if SaveData==1
    
    SavePath = uigetdir('C:\','Select Folder to Save FROG Reconstruction');
%     mkdir('C:\Users\hrt\Desktop\Data\Carlson_FROG\Trace32\','FROG_Reconstruction')
    cd(SavePath)
    
    dlmwrite('ExperimentalFROG.txt',SExp,'delimiter','\t')
    dlmwrite('ReconstructedFROG.txt',Srecon,'delimiter','\t')
    dlmwrite('AngularFrequency.txt',OmegaExp','delimiter','\t')
    dlmwrite('Delay.txt',DelayExp','delimiter','\t')
    
    dlmwrite('ReconstructedPulseTemporal.txt',ReconstructedT,'delimiter','\t')
    dlmwrite('ReconstructedPulseSpectrum.txt',ReconstructedW,'delimiter','\t')
    
end

