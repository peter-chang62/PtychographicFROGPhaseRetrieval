figure;
subplot(1,2,2)
surf(Delay,AngularFrequency,ReconstructedFROG)
view(2)
shading flat
axis([Delay(1) Delay(end) AngularFrequency(1) AngularFrequency(end)])
colormap(jet)
title('Reconstructed Frog')
xlabel('Time (fs)')
ylabel('Frequency (rad/fs)')



subplot(1,2,1)
surf(Delay,AngularFrequency,ExperimentalFROG)
view(2)
shading flat
axis([Delay(1) Delay(end) AngularFrequency(1) AngularFrequency(end)])
colormap(jet)
title('Experimental Frog')
xlabel('Time (fs)')
ylabel('Frequency (rad/fs)')
%%
figure;
subplot(1,2,2)
yyaxis left
plot(ReconstructedPulseSpectrum(:,3),ReconstructedPulseSpectrum(:,1))
ylabel('Amplitude')
yyaxis right
plot(ReconstructedPulseSpectrum(:,3),ReconstructedPulseSpectrum(:,2))
ylabel('Phase (rad)')
xlabel('wavelength (\mum)')
title('Temporal Reconstruction')
%xlim([1.250 1.800])
xlim([1 2])
subplot(1,2,1)
plot(ReconstructedPulseTemporal(:,3),ReconstructedPulseTemporal(:,1).^2)
axis([-1000 1000 -0 1.025])
title('Temporal Reconstruction')
xlabel('Time (fs)')
ylabel('Amplitude')


%%
figure;
semilogy(Untitled(:,1),Untitled(:,2),ReconstructedPulseSpectrum(:,3),ReconstructedPulseSpectrum(:,1)/max(ReconstructedPulseSpectrum(:,1)))
ylim([10^-3 1])
xlim([1.3,1.8])

legend('OSA Trace','Frog Reconstruction')
ylabel('Power')
xlabel('wavelength (\mum)')

%%
figure;

semilogy(Untitled(:,1),Untitled(:,2),SSpec(:,1),SSpec(:,2))
ylim([10^-3 1])
xlim([1.2,1.9])

legend('OSA Trace','Simulation')
ylabel('Power')
xlabel('wavelength (\mum)')