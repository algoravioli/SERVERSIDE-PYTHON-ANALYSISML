function [s,f,t] = compute_stft(x,window,noverlap,fs)
%COMPUTE_STFT Computes one-sided spectrogram of signal x based on window,
%number of overlap samples and sampling frequency fs. NFFT length is same
%as window size i.e., no zero padding.

%% column vector checks
if isrow(x)
    x = x';
end

if isrow(window)
    window = window';
end

%% Set up frequency, time, window and loop parameters

N = length(window); % NFFT length

% one-sided frequency bin list
df = fs/N;          % frequency bin size
f = (0:df:fs/2)';   % one-sided frequency bin list
fLen = length(f);   % length of frequency bin list

% initial time parameter
t = N/2/fs;      % centre time of first window

% initial window parameters
winStart = 1;
winEnd = N;
Nsignal = length(x);

% initial loop parameters
index = 1;
s = zeros(fLen,1);

%% Compute STFT
while winEnd <= Nsignal % stop if window end exceeds signal length

    % update time step list (only after first iteration)
    if index>1
        tc = (winStart + winEnd - 1)/2/fs; % centre time of new window position
        t = [t,tc];
    end

    xWin = x(winStart:winEnd).*window; % window part of signal
    
    X = fft(xWin);    % compute FFT
    X = X(1:fLen);    % take only positive frequencies

    % update spectrogram amplitude
    s = [s,X];

    % update window start and end points
    winStart = winStart + (N - noverlap);
    winEnd = winEnd + (N - noverlap);
    index = index+1;

end

s = s(:,2:end); % remove first column of zeros

end
