import React, { useMemo, useState } from 'react';
import { useImageStore } from '../../store/imageStore';
import { exportRoadNetworks } from '../../services/api';
import { CoordinateFormat, ExportOptions, DeleteOptions, DeleteResponseMeta } from '../../types';
import ExportNotification from '../common/ExportNotification';

interface ExportDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onExportDone?: (meta: { exported: number; missing: number; filename?: string }) => void;
}

const ExportDialog: React.FC<ExportDialogProps> = ({ isOpen, onClose, onExportDone }) => {
  const currentImage = useImageStore((s) => s.currentImage);
  const [scope, setScope] = useState<'current' | 'all'>('current');
  const [coord, setCoord] = useState<CoordinateFormat>('xy');
  const [isExporting, setIsExporting] = useState(false);
  const [notification, setNotification] = useState<string | null>(null);

  const canExport = useMemo(() => {
    return scope === 'all' || (scope === 'current' && !!currentImage);
  }, [scope, currentImage]);

  const handleStartExport = async () => {
    if (!canExport || isExporting) return;
    setIsExporting(true);
    try {
      const options: ExportOptions = {
        scope,
        imageId: scope === 'current' ? currentImage?.id : undefined,
        coordinateFormat: coord,
      };
      const { blob, meta } = await exportRoadNetworks(options);
      // Decide whether to download based on results
      const nothingToExport = meta.exported === 0
      if ((scope === 'current' && nothingToExport) || (scope === 'all' && nothingToExport)) {
        // Do not download, show message
        setNotification(`Successfully exported ${meta.exported} result(s), ${meta.missing} without annotations.`)
        return
      }

      // Proceed with download
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const fallbackName = scope === 'all' ? 'all.zip' : `${(currentImage?.original_filename || currentImage?.name || 'export').replace(/\.[^/.]+$/, '')}.pickle`
      a.download = meta.filename || fallbackName
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)

      // Message + optional toast
      setNotification(`Successfully exported ${meta.exported} result(s), ${meta.missing} without annotations.`)
      if (scope === 'all' && onExportDone) onExportDone(meta)

      // Auto-close after success
      setTimeout(() => onClose(), 2500)
    } catch (err: any) {
      setNotification(`Export failed: ${err?.message || 'Unknown error'}`);
    } finally {
      setIsExporting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative bg-white text-gray-900 rounded-2xl shadow-2xl w-[420px] p-6 border border-gray-200">
        <div className="text-xl font-semibold mb-4 text-center text-gray-800">Export Road Networks</div>
        <div className="space-y-5">
          <div>
            <div className="text-sm font-medium mb-2 text-gray-700">Scope</div>
            <div className="grid grid-cols-2 gap-2">
              <label className={`group cursor-pointer rounded-lg border px-3 py-2 flex items-center gap-2 ${scope==='current' ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}> 
                <input className="accent-blue-600" type="radio" name="scope" checked={scope==='current'} onChange={() => setScope('current')} />
                <span>Current only</span>
              </label>
              <label className={`group cursor-pointer rounded-lg border px-3 py-2 flex items-center gap-2 ${scope==='all' ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}> 
                <input className="accent-blue-600" type="radio" name="scope" checked={scope==='all'} onChange={() => setScope('all')} />
                <span>All</span>
              </label>
            </div>
          </div>
          <div>
            <div className="text-sm font-medium mb-2 text-gray-700">Coordinate format</div>
            <div className="grid grid-cols-2 gap-2">
              <label className={`group cursor-pointer rounded-lg border px-3 py-2 flex items-center gap-2 ${coord==='rc' ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}>
                <input className="accent-blue-600" type="radio" name="coord" checked={coord==='rc'} onChange={() => setCoord('rc')} />
                <span>(r, c)</span>
              </label>
              <label className={`group cursor-pointer rounded-lg border px-3 py-2 flex items-center gap-2 ${coord==='xy' ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}>
                <input className="accent-blue-600" type="radio" name="coord" checked={coord==='xy'} onChange={() => setCoord('xy')} />
                <span>(x, y)</span>
              </label>
            </div>
          </div>
          {notification && (
            <div className="text-xs text-gray-700 bg-gray-50 rounded-md p-3 border border-gray-200 text-center">{notification}</div>
          )}
          <div className="flex gap-3 justify-end pt-1">
            <button
              onClick={onClose}
              className="px-4 h-9 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 shadow-sm"
            >Cancel</button>
            <button
              onClick={handleStartExport}
              disabled={!canExport || isExporting}
              className={`px-4 h-9 rounded-lg text-white shadow-sm ${canExport && !isExporting ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-300 cursor-not-allowed'}`}
            >{isExporting ? 'Exporting...' : 'Export'}</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export const ToolBar: React.FC = () => {
  const [isDialogOpen, setDialogOpen] = useState(false);
  const [isDeleteOpen, setDeleteOpen] = useState(false);
  const [toast, setToast] = useState<{exported: number, missing: number, filename?: string} | null>(null)
  const deleteImages = useImageStore(s => s.deleteImages);
  const currentImage = useImageStore(s => s.currentImage);

  // Delete Dialog local state
  const [delScope, setDelScope] = useState<'current' | 'all'>('current');
  const [deleteAnno, setDeleteAnno] = useState<boolean>(true);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteMsg, setDeleteMsg] = useState<string | null>(null);

  return (
    <aside className="w-[60px] bg-gray-100 h-full flex flex-col items-center py-4 space-y-4 border-r border-gray-200">
      {/* Delete Button (top-most) */}
      <button
        onClick={() => setDeleteOpen(true)}
        className="w-[34px] h-[34px] rounded bg-fuchsia-500 text-white text-sm flex items-center justify-center hover:bg-red-700"
        title="Delete"
        aria-label="Delete"
      >
        ✖
      </button>

      <button
        onClick={() => setDialogOpen(true)}
        className="w-[34px] h-[34px] rounded bg-amber-700 text-white text-sm flex items-center justify-center hover:bg-blue-800"
        title="Export"
        aria-label="Export"
      >
        ⇩
      </button>
      <ExportDialog isOpen={isDialogOpen} onClose={() => setDialogOpen(false)} onExportDone={(meta) => setToast(meta)} />
      {toast && (
        <ExportNotification exported={toast.exported} missing={toast.missing} filename={toast.filename} onClose={() => setToast(null)} />
      )}

      {/* Delete Dialog */}
      {isDeleteOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/40" onClick={() => setDeleteOpen(false)} />
          <div className="relative bg-white text-gray-900 rounded-2xl shadow-2xl w-[420px] p-6 border border-gray-200">
            <div className="text-xl font-semibold mb-4 text-center text-gray-800">Delete Images</div>
            <div className="space-y-5">
              <div>
                <div className="text-sm font-medium mb-2 text-gray-700">Scope</div>
                <div className="grid grid-cols-2 gap-2">
                  <label className={`group cursor-pointer rounded-lg border px-3 py-2 flex items-center gap-2 ${delScope==='current' ? 'border-red-500 bg-red-50' : 'border-gray-300 hover:border-gray-400'}`}> 
                    <input className="accent-red-600" type="radio" name="delscope" checked={delScope==='current'} onChange={() => setDelScope('current')} />
                    <span>Current only</span>
                  </label>
                  <label className={`group cursor-pointer rounded-lg border px-3 py-2 flex items-center gap-2 ${delScope==='all' ? 'border-red-500 bg-red-50' : 'border-gray-300 hover:border-gray-400'}`}> 
                    <input className="accent-red-600" type="radio" name="delscope" checked={delScope==='all'} onChange={() => setDelScope('all')} />
                    <span>All</span>
                  </label>
                </div>
              </div>
              <div>
                <div className="text-sm font-medium mb-2 text-gray-700">Also delete annotations</div>
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input type="checkbox" className="accent-red-600" checked={deleteAnno} onChange={(e) => setDeleteAnno(e.target.checked)} />
                  <span>Delete annotation files</span>
                </label>
              </div>
              {deleteMsg && (
                <div className="text-xs text-gray-700 bg-gray-50 rounded-md p-3 border border-gray-200 text-center">{deleteMsg}</div>
              )}
              <div className="flex gap-3 justify-end pt-1">
                <button
                  onClick={() => setDeleteOpen(false)}
                  className="px-4 h-9 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 shadow-sm"
                >Cancel</button>
                <button
                  onClick={async () => {
                    if (isDeleting) return;
                    setIsDeleting(true);
                    setDeleteMsg(null);
                    try {
                      const opts: DeleteOptions = {
                        scope: delScope,
                        imageId: delScope === 'current' ? currentImage?.id : undefined,
                        deleteAnnotations: deleteAnno,
                      };
                      const res: DeleteResponseMeta = await deleteImages(opts);
                      setDeleteMsg(`Deleted ${res.deleted}, skipped ${res.skipped}.`);
                      // Auto-close shortly after success
                      setTimeout(() => setDeleteOpen(false), 1200);
                    } catch (err: any) {
                      setDeleteMsg(`Delete failed: ${err?.message || 'Unknown error'}`);
                    } finally {
                      setIsDeleting(false);
                    }
                  }}
                  className={`px-4 h-9 rounded-lg text-white shadow-sm ${!isDeleting ? 'bg-red-600 hover:bg-red-700' : 'bg-red-300 cursor-not-allowed'}`}
                >{isDeleting ? 'Deleting...' : 'Delete'}</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
};