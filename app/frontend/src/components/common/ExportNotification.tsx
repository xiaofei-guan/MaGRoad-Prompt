interface ExportNotificationProps {
  exported: number
  missing: number
  filename?: string
  onClose: () => void
  durationMs?: number
}

/**
 * A lightweight toast-style notification for export completion.
 * Renders top-right, auto-dismisses after `durationMs`.
 */
export default function ExportNotification({ exported, missing, filename, onClose, durationMs = 2500 }: ExportNotificationProps) {
  // Auto-dismiss simple variant to avoid extra effects
  setTimeout(onClose, durationMs)

  return (
    <div className="fixed top-4 right-4 z-[100]">
      <div className="bg-white text-gray-900 rounded-lg shadow-xl border border-gray-200 w-[320px] overflow-hidden">
        <div className="px-4 py-3 bg-blue-600 text-white font-semibold">Export completed</div>
        <div className="px-4 py-3 space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Exported</span>
            <span className="font-medium">{exported}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Without annotations</span>
            <span className="font-medium">{missing}</span>
          </div>
          {filename && (
            <div className="pt-1 text-xs text-gray-500 truncate" title={filename}>File: {filename}</div>
          )}
        </div>
        <div className="px-4 py-2 bg-gray-50 flex justify-end">
          <button onClick={onClose} className="h-8 px-3 rounded bg-gray-200 hover:bg-gray-300 text-gray-800 text-sm">Close</button>
        </div>
      </div>
    </div>
  )
}


