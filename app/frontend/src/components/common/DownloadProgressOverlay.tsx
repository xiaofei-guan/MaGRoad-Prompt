import React from 'react';

interface DownloadProgressOverlayProps {
	visible: boolean;
	percentage: number; // 0-100
	downloadSpeedBytesPerSec?: number;
	estimatedTimeRemainingSec?: number;
	message?: string;
}

function formatSpeed(bytesPerSec?: number): string {
	if (!bytesPerSec || bytesPerSec <= 0) return '';
	const mbps = bytesPerSec / 1024 / 1024;
	return `${mbps.toFixed(1)} MB/s`;
}

function formatETA(seconds?: number): string {
	if (seconds === undefined || seconds <= 0) return '';
	if (seconds < 60) return `${Math.round(seconds)}s`;
	const m = Math.floor(seconds / 60);
	const s = Math.round(seconds % 60);
	return `${m}m ${s}s`;
}

const DownloadProgressOverlay: React.FC<DownloadProgressOverlayProps> = ({
	visible,
	percentage,
	downloadSpeedBytesPerSec,
	estimatedTimeRemainingSec,
	message
}) => {
	if (!visible) return null;
	const speed = formatSpeed(downloadSpeedBytesPerSec);
	const eta = formatETA(estimatedTimeRemainingSec);
	return (
		<div className="absolute inset-0 flex items-center justify-center z-50 pointer-events-none">
			<div className="bg-white/95 backdrop-blur-md border border-gray-200/50 shadow-2xl rounded-2xl px-6 py-5 max-w-sm w-80 mx-4 text-gray-800">
				<div className="text-center mb-3">
					<div className="text-lg font-semibold">{message || 'Loading Image'}</div>
				</div>
				<div className="text-center mb-3">
					<div className="text-2xl font-bold">{Math.round(percentage)}%</div>
				</div>
				<div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden shadow-inner mb-3">
					<div
						className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full transition-all duration-300 ease-out"
						style={{ width: `${Math.max(0, Math.min(100, percentage))}%` }}
					/>
				</div>
				<div className="flex items-center justify-between text-xs text-gray-600">
					<div>{speed}</div>
					<div>{eta ? `${eta} left` : ''}</div>
				</div>
			</div>
		</div>
	);
};

export default DownloadProgressOverlay;


