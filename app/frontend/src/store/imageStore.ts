import { create } from 'zustand';
import { ImageInfo, FeatureStatus, UploadResponse, FeatureStatusResponse, PrecomputeResponse, DeleteOptions, DeleteResponseMeta } from '../types';
import { imageApi } from '../services/api'; // Import real API service

// Natural, locale-aware, numeric file-name sort collator
const humanNameCollator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

// Get a comparable display name for sorting (prefer original filename)
function getComparableName(image: ImageInfo): string {
  // Lowercase to ensure case-insensitive ordering
  return (image.original_filename || image.name || image.filename || image.id || '').toLowerCase();
}

// Sort images from small to large by name using natural numeric ordering
function sortImagesByNameAscending(images: ImageInfo[]): ImageInfo[] {
  return [...images].sort((a, b) => humanNameCollator.compare(getComparableName(a), getComparableName(b)));
}

interface ImageState {
  availableImages: ImageInfo[];
  currentImage: ImageInfo | null;
  featureStatus: FeatureStatus;
  isFeatureLoading: boolean; // For feature computation
  isUploading: boolean; // For file upload
  uploadProgress: {
    totalFiles: number;
    uploadedFiles: number;
    totalBytes: number;
    uploadedBytes: number;
    percent: number;
    currentFile?: string;
  } | null;
  error: string | null;
  isStartingFeatureOp: boolean; // New flag

  setAvailableImages: (images: ImageInfo[]) => void;
  setCurrentImageById: (imageId: string) => void;
  updateCurrentImage: (updates: Partial<ImageInfo>) => void; // New function to update current image
  setFeatureStatus: (status: FeatureStatus) => void;
  startFeatureComputation: (imageId: string) => Promise<void>; // Uses real API
  recomputeImageFeatures: (imageId: string) => Promise<void>;
  fetchAvailableImages: () => Promise<void>; // Uses real API
  fetchImagesForceRefresh: () => Promise<void>; // New function to bypass the upload check
  uploadAndRefreshImages: (files: File[]) => Promise<void>; // New action combining upload + refresh
  deleteImages: (options: DeleteOptions) => Promise<DeleteResponseMeta>;
  setIsUploading: (uploading: boolean) => void;
  // Navigation functions for previous/next image
  switchToPreviousImage: () => void;
  switchToNextImage: () => void;
}

const POLLING_INTERVAL = 500; // Check status every 0.5 seconds
const MAX_POLLING_ATTEMPTS = 60; // Stop polling after 60 seconds (60 * 0.5s)

// Helper to map backend status to frontend FeatureStatus
const mapBackendStatusToFeatureStatus = (backendStatus: FeatureStatusResponse['status']): FeatureStatus => {
    switch (backendStatus) {
        case 'none': return 'none';
        case 'completed': return 'ready';
        case 'ready': return 'ready'; // Explicitly handle 'ready' status from backend
        case 'computing': return 'computing'; // Handle large image computing status
        case 'processing': return 'computing';
        case 'pending': return 'computing'; // Or a new 'pending' state if distinct UI needed
        case 'failed': return 'error';
        case 'error': return 'error'; // Handle error status from backend
        default: return 'none'; // Should not happen with typed responses
    }
};

export const useImageStore = create<ImageState>((set, get) => ({
  availableImages: [],
  currentImage: null,
  featureStatus: 'none',
  isFeatureLoading: false,
  isUploading: false,
  uploadProgress: null,
  error: null,
  isStartingFeatureOp: false, // Initialize new flag

  // Always keep availableImages sorted by natural filename order (ascending)
  setAvailableImages: (images) => set({ availableImages: sortImagesByNameAscending(images) }),

  setCurrentImageById: (imageId) => {
    const image = get().availableImages.find(img => img.id === imageId);
    if (image) {
        console.log("Setting current image:", image);
        
        // Make sure we have the latest annotation status directly from the backend
        // This will ensure the status is always accurate, even after a server restart
        (async () => {
            try {
                // Force a refresh of all images to get the latest status from the server
                await get().fetchImagesForceRefresh();
                
                // After refresh, get the updated image data
                const updatedImageList = get().availableImages;
                const updatedImage = updatedImageList.find(img => img.id === imageId);
                
                if (updatedImage) {
                    console.log("Updated current image with latest status:", updatedImage);
                    set({
                        currentImage: updatedImage,
                        featureStatus: updatedImage.features_computed ? 'ready' : 'none',
                        error: null
                    });
                }
            } catch (err) {
                console.error("Error updating image status:", err);
                // Fall back to using the existing image data if the refresh fails
                set({
                    currentImage: image,
                    featureStatus: image.features_computed ? 'ready' : 'none',
                    error: null
                });
            }
        })();
    } else {
      console.warn(`Image with ID ${imageId} not found in available images.`);
      set({ currentImage: null, featureStatus: 'none' });
    }
  },

  updateCurrentImage: (updates) => {
    const current = get().currentImage;
    if (!current) return;
    
    console.log("Updating current image with:", updates);
    set({
      currentImage: { ...current, ...updates }
    });
  },

  setFeatureStatus: (status) => set({ featureStatus: status, isFeatureLoading: status === 'computing' }),
  setIsUploading: (uploading) => set({ isUploading: uploading }),

  // --- Actions using real API ---

  uploadAndRefreshImages: async (files) => {
    // Concurrency-limited per-file uploads with progress aggregation
    const CONCURRENCY_LIMIT = 3;
    const fileDescriptors = files.map((file, index) => ({ id: `${index}-${file.name}-${file.size}`, file }));
    const totalBytes = fileDescriptors.reduce((sum, f) => sum + f.file.size, 0);
    const perFileLoaded: Record<string, number> = {};
    let uploadedFiles = 0;
    let firstUploadedImageId: string | null = null;

    const updateProgress = (currentFileName?: string) => {
      const uploadedBytes = Object.values(perFileLoaded).reduce((a, b) => a + b, 0);
      const percent = totalBytes > 0 ? Math.min(100, Math.round((uploadedBytes / totalBytes) * 100)) : 0;
      set({ uploadProgress: { totalFiles: fileDescriptors.length, uploadedFiles, totalBytes, uploadedBytes, percent, currentFile: currentFileName } });
    };

    set({ isUploading: true, error: null, uploadProgress: { totalFiles: fileDescriptors.length, uploadedFiles: 0, totalBytes, uploadedBytes: 0, percent: 0 } });

    try {
      let cursor = 0;
      const runNext = async (): Promise<void> => {
        const myIndex = cursor++;
        const item = fileDescriptors[myIndex];
        if (!item) return;
        perFileLoaded[item.id] = 0;
        updateProgress(item.file.name);
        try {
          const resp: UploadResponse = await imageApi.uploadSingleImage(item.file, (loaded, _total) => {
            perFileLoaded[item.id] = loaded;
            updateProgress(item.file.name);
          });
          uploadedFiles += 1;
          updateProgress();
          if (!firstUploadedImageId && resp.images && resp.images.length > 0) {
            firstUploadedImageId = resp.images[0].id;
          }
        } catch (error) {
          console.error('Error uploading file:', item.file.name, error);
          // Continue with others; set error but don't abort
          set({ error: error instanceof Error ? error.message : 'Upload failed for some files' });
        }
        // Recurse to process next item in this worker
        await runNext();
      };

      // Start workers
      const workers: Promise<void>[] = [];
      for (let i = 0; i < Math.min(CONCURRENCY_LIMIT, fileDescriptors.length); i++) {
        workers.push(runNext());
      }
      await Promise.all(workers);

      // Refresh the image list
      await get().fetchImagesForceRefresh();

      if (firstUploadedImageId) {
        get().setCurrentImageById(firstUploadedImageId);
      }
    } catch (err) {
      console.error('Upload and refresh error:', err);
      set({ error: err instanceof Error ? err.message : 'Upload failed' });
      try {
        await get().fetchImagesForceRefresh();
      } catch (fetchErr) {
        console.error('Failed to refresh images after upload error:', fetchErr);
      }
    } finally {
      set({ isUploading: false, uploadProgress: null });
    }
  },

  // Delete images API wrapper with auto refresh and selection handling
  deleteImages: async (options) => {
    try {
      const currentId = get().currentImage?.id;
      const meta = await imageApi.deleteImages(options);
      await get().fetchImagesForceRefresh();

      // If we deleted current, ensure selection points to first item if exists
      if (options.scope === 'current' && currentId) {
        const stillThere = get().availableImages.find(img => img.id === currentId);
        if (!stillThere && get().availableImages.length > 0) {
          get().setCurrentImageById(get().availableImages[0].id);
        }
      }
      return meta;
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Delete failed' });
      throw error;
    }
  },

  fetchAvailableImages: async () => {
    // Simplified guard: let ongoing feature computation polling manage its own state.
    // Avoid fetching if actively uploading.
    if (get().isUploading) {
        console.log("Skipping fetchAvailableImages due to ongoing upload.");
        return;
    }
    
    await get().fetchImagesForceRefresh();
  },

  // New function to fetch images without checking isUploading
  fetchImagesForceRefresh: async () => {
    set({ error: null }); // Clear previous errors on new fetch attempt
    try {
      console.log("Fetching available images...");
      const response = await imageApi.listImages();
      const imagesUnsorted: ImageInfo[] = response.images.map((imgAPI: any) => {
        console.log(`Image ${imgAPI.id} (${imgAPI.original_filename}): is_annotated=${imgAPI.is_annotated}`);
        return {
          id: imgAPI.id,
          url: imgAPI.url,
          filename: imgAPI.filename,
          original_filename: imgAPI.original_filename,
          name: imgAPI.original_filename, // Use original filename for display
          hasFeature: imgAPI.has_feature,
          features_computed: imgAPI.has_feature, // Keep this mapping
          isAnnotated: imgAPI.is_annotated || false, // Default to false if not provided
        };
      });
      const images = sortImagesByNameAscending(imagesUnsorted);
      set({ availableImages: images });
      console.log(`Fetched ${images.length} images from server (sorted ascending by name).`);

      // Log annotation status of all images for debugging
      console.log("Annotation status summary:");
      images.forEach(img => {
        console.log(`- ${img.id} (${img.original_filename}): isAnnotated=${img.isAnnotated}`);
      });

      const currentId = get().currentImage?.id;
      if (currentId) {
          const currentImgData = images.find((img: ImageInfo) => img.id === currentId);
          if (currentImgData) {
              // If current image data is updated (e.g., features_computed changed), reflect it
              if (get().currentImage?.features_computed !== currentImgData.features_computed ||
                  get().currentImage?.isAnnotated !== currentImgData.isAnnotated) {
                  console.log(`Updating current image ${currentId} status from fetch.`);
              }
              set({
                  currentImage: currentImgData,
                  // Update featureStatus based on the fresh data for the current image
                  featureStatus: currentImgData.features_computed ? 'ready' : (get().featureStatus === 'computing' && get().currentImage?.id === currentId ? 'computing' : 'none')
              });
          } else {
              console.warn(`Current image ${currentId} no longer available after refresh. Clearing selection.`);
              set({ currentImage: null, featureStatus: 'none' });
          }
      } else if (images.length > 0) { // No current image, but images are available
           console.log("Auto-selecting first image after fetch.");
           get().setCurrentImageById(images[0].id);
      }

    } catch (err) {
      console.error("Error fetching images:", err);
      set({ error: err instanceof Error ? err.message : 'Failed to load images' });
    }
  },

  startFeatureComputation: async (imageId) => {
    if (get().isStartingFeatureOp) {
      console.warn("Feature operation (startFeatureComputation) is already in progress. Skipping duplicate call.");
      return;
    }
    set({ isStartingFeatureOp: true });

    try {
      const currentStoreState = get();
      const { currentImage, featureStatus: currentGlobalFeatureStatus, availableImages, isUploading } = currentStoreState;

      if (currentImage?.id !== imageId) {
        console.warn(`startFeatureComputation called for image ${imageId}, but current selected image is ${currentImage?.id}. Skipping.`);
        set({ isStartingFeatureOp: false });
        return;
      }

      const imageInData = availableImages.find(img => img.id === imageId);
      if (imageInData?.features_computed) {
        console.warn(`Image ${imageId} data indicates features already computed. Ensuring status is 'ready'.`);
        set(state => ({
            featureStatus: 'ready',
            isFeatureLoading: false,
            currentImage: state.currentImage && state.currentImage.id === imageId ? { ...state.currentImage, features_computed: true, hasFeature: true } : state.currentImage,
            isStartingFeatureOp: false
        }));
        return;
      }

      if (currentGlobalFeatureStatus === 'computing' && currentImage?.id === imageId) {
        console.warn(`Feature computation for ${imageId} is already in progress. Skipping.`);
        set({ isStartingFeatureOp: false });
        return;
      }
      if (currentGlobalFeatureStatus === 'ready' && currentImage?.id === imageId) {
        console.warn(`Features for ${imageId} are already ready. Skipping.`);
        set({ isStartingFeatureOp: false });
        return;
      }
    
      if (isUploading) {
        console.warn("Upload in progress, skipping feature computation trigger.");
        set({ isStartingFeatureOp: false });
        return;
      }

      set({ isFeatureLoading: true, featureStatus: 'computing', error: null });

      // call API to check if features already exist
      try {
        // maybe features already computed, but the frontend status is wrong
        const statusResponse = await imageApi.checkFeatureStatus(imageId);
        if (statusResponse.status === 'completed' || statusResponse.status === 'ready') {
          console.log(`Features for ${imageId} already computed according to status check. Setting status to ready.`);
          const updatedAvailableImages = get().availableImages.map(img =>
            img.id === imageId ? { ...img, features_computed: true, hasFeature: true } : img
          );
          const updatedCurrentImage = updatedAvailableImages.find(img => img.id === imageId);
          set({
            availableImages: updatedAvailableImages,
            currentImage: updatedCurrentImage || get().currentImage,
            featureStatus: 'ready',
            isFeatureLoading: false,
            error: null,
            isStartingFeatureOp: false
          });
          return;
        }
      } catch (statusCheckErr) {
        // if the status check API call fails, we still continue to try to start feature computation
        console.warn("Error checking initial feature status:", statusCheckErr);
      }

      let pollingTimeoutId: NodeJS.Timeout | null = null;
      let attempts = 0;

      const pollStatus = async () => {
        if (pollingTimeoutId) clearTimeout(pollingTimeoutId);
        
        const latestState = get();
        if (latestState.currentImage?.id !== imageId || latestState.featureStatus !== 'computing') {
            if (latestState.featureStatus === 'computing' && latestState.currentImage?.id === imageId) {
                // This case should ideally not be hit if logic is sound
            } else if (latestState.featureStatus !== 'computing' && latestState.isFeatureLoading) {
                set({ isFeatureLoading: false });
            }
            return;
        }

        if (attempts >= MAX_POLLING_ATTEMPTS) {
            set({ isFeatureLoading: false, featureStatus: 'error', error: 'Feature computation timeout', isStartingFeatureOp: false });
            return;
        }
        attempts++;
        
        try {
            const statusResponse = await imageApi.checkFeatureStatus(imageId);
            const afterAwaitState = get();
            if (afterAwaitState.currentImage?.id !== imageId || afterAwaitState.featureStatus !== 'computing') {
                if (afterAwaitState.featureStatus !== 'computing' && afterAwaitState.isFeatureLoading) 
                    set({isFeatureLoading: false});
                return;
            }

            // Get the correctly mapped frontend status
            const mappedStatus = mapBackendStatusToFeatureStatus(statusResponse.status);
            console.log(`Polling attempt ${attempts} for ${imageId}, status: ${statusResponse.status}, mapped status: ${mappedStatus}`);
            
            if (mappedStatus === 'computing') {
                // Continue polling
                pollingTimeoutId = setTimeout(pollStatus, POLLING_INTERVAL);
            } else if (mappedStatus === 'ready') {
                // Features are ready - update all state in one operation
                const updatedAvailableImgs = get().availableImages.map(img =>
                    img.id === imageId ? { ...img, features_computed: true, hasFeature: true } : img
                );
                const updatedCurrentImg = updatedAvailableImgs.find(img => img.id === imageId);
                
                set({
                    availableImages: updatedAvailableImgs,
                    currentImage: updatedCurrentImg || get().currentImage,
                    featureStatus: 'ready',
                    isFeatureLoading: false,
                    error: null,
                    isStartingFeatureOp: false
                });
                
                console.log(`Polling successful for ${imageId}, features are ready.`);
            } else {
                // Handle error or other states
                set({
                    featureStatus: mappedStatus,
                    isFeatureLoading: false,
                    isStartingFeatureOp: false,
                    error: mappedStatus === 'error' ? (statusResponse.error || 'Feature computation failed') : null
                });
            }
        } catch (err) {
            console.error("Polling error:", err);
            const afterErrorState = get();
            if (attempts < MAX_POLLING_ATTEMPTS && afterErrorState.currentImage?.id === imageId && afterErrorState.featureStatus === 'computing'){
                pollingTimeoutId = setTimeout(pollStatus, POLLING_INTERVAL * 2); // Longer delay on error before retry
            } else if (afterErrorState.currentImage?.id === imageId && afterErrorState.featureStatus === 'computing') {
                set({
                  isFeatureLoading: false, 
                  featureStatus: 'error', 
                  error: 'Polling failed after multiple retries', 
                  isStartingFeatureOp: false
                });
            } else if (afterErrorState.isFeatureLoading) {
                set({
                  isFeatureLoading: false,
                  isStartingFeatureOp: false
                }); 
            }
        }
      }; // End of pollStatus

      // Now we try to start the computation (only if we didn't already find it completed above)
      try {
        const precomputeResponse: PrecomputeResponse = await imageApi.startPrecomputeFeature(imageId);
        console.log(`Precompute API response for ${imageId}: ${precomputeResponse.message}`);
      
        const stateAfterPrecomputeCall = get();
        if (stateAfterPrecomputeCall.currentImage?.id !== imageId || stateAfterPrecomputeCall.featureStatus !== 'computing') {
            console.warn("Image selection or feature status changed during/after precompute API call response. Aborting.");
            if(stateAfterPrecomputeCall.featureStatus !== 'computing' && stateAfterPrecomputeCall.isFeatureLoading) {
                set({isFeatureLoading: false});
            }
            return;
        }

        // Check if features already exist based on the response message
        if (precomputeResponse.message.includes("Features already exist")) {
            console.log(`Features for ${imageId} already exist according to precompute API. Setting status to ready.`);
            const updatedAvailableImages = get().availableImages.map(img =>
                img.id === imageId ? { ...img, features_computed: true, hasFeature: true } : img
            );
            const updatedCurrentImage = updatedAvailableImages.find(img => img.id === imageId);
            set({
                availableImages: updatedAvailableImages,
                currentImage: updatedCurrentImage || get().currentImage,
                featureStatus: 'ready',
                isFeatureLoading: false,
                error: null,
                isStartingFeatureOp: false
            });
        } else {
            // Features need to be computed, start polling
            console.log(`Starting computation polling for ${imageId}`);
            pollingTimeoutId = setTimeout(pollStatus, POLLING_INTERVAL / 2);
        }
      } catch (err) {
        console.error(`Error starting feature computation for ${imageId}:`, err);
        const stateOnStartError = get();
        if (stateOnStartError.currentImage?.id === imageId && stateOnStartError.featureStatus === 'computing') {
          set({ 
            featureStatus: 'error', 
            isFeatureLoading: false, 
            error: err instanceof Error ? err.message : 'Failed to start feature computation',
            isStartingFeatureOp: false 
          });
        } else {
           console.warn("Error occurred during feature computation initiation, but current context has changed. Not setting global error state.");
           if (stateOnStartError.isFeatureLoading) set({isFeatureLoading: false});
        }
      }
    } finally {
      // We only set isStartingFeatureOp to false here if we didn't already set it in one of the early returns
      // This avoids race conditions where multiple concurrent state updates could override each other
      const finalState = get();
      if (finalState.isStartingFeatureOp) {
        set({ isStartingFeatureOp: false });
      }
    }
  },

  // Force re-computation of features even if they already exist
  recomputeImageFeatures: async (imageId) => {
    if (get().isStartingFeatureOp) {
      console.warn("Feature operation (recomputeImageFeatures) is already in progress. Skipping duplicate call.");
      return;
    }
    set({ isStartingFeatureOp: true });

    try {
      const { currentImage, isUploading } = get();
      if (currentImage?.id !== imageId) {
        console.warn(`recomputeImageFeatures called for image ${imageId}, but current selected image is ${currentImage?.id}. Skipping.`);
        set({ isStartingFeatureOp: false });
        return;
      }
      if (isUploading) {
        console.warn("Upload in progress, skipping feature recomputation trigger.");
        set({ isStartingFeatureOp: false });
        return;
      }

      // Set to computing state immediately
      set({ isFeatureLoading: true, featureStatus: 'computing', error: null });

      // Start recomputation on backend
      await imageApi.recomputeFeatures(imageId);

      let pollingTimeoutId: NodeJS.Timeout | null = null;
      let attempts = 0;

      const pollStatus = async () => {
        if (pollingTimeoutId) clearTimeout(pollingTimeoutId);

        const latestState = get();
        if (latestState.currentImage?.id !== imageId || latestState.featureStatus !== 'computing') {
          if (latestState.featureStatus !== 'computing' && latestState.isFeatureLoading) set({ isFeatureLoading: false });
          return;
        }

        if (attempts >= MAX_POLLING_ATTEMPTS) {
          set({ isFeatureLoading: false, featureStatus: 'error', error: 'Feature computation timeout', isStartingFeatureOp: false });
          return;
        }
        attempts++;

        try {
          const statusResponse = await imageApi.checkFeatureStatus(imageId);
          const mappedStatus = mapBackendStatusToFeatureStatus(statusResponse.status);
          if (mappedStatus === 'computing') {
            pollingTimeoutId = setTimeout(pollStatus, POLLING_INTERVAL);
          } else if (mappedStatus === 'ready') {
            const updatedAvailableImgs = get().availableImages.map(img => img.id === imageId ? { ...img, features_computed: true, hasFeature: true } : img);
            const updatedCurrentImg = updatedAvailableImgs.find(img => img.id === imageId);
            set({
              availableImages: updatedAvailableImgs,
              currentImage: updatedCurrentImg || get().currentImage,
              featureStatus: 'ready',
              isFeatureLoading: false,
              error: null,
              isStartingFeatureOp: false
            });
          } else {
            set({ featureStatus: mappedStatus, isFeatureLoading: false, isStartingFeatureOp: false, error: mappedStatus === 'error' ? (statusResponse.error || 'Feature computation failed') : null });
          }
        } catch (err) {
          console.error("Polling error (recompute):", err);
          const afterErrorState = get();
          if (attempts < MAX_POLLING_ATTEMPTS && afterErrorState.currentImage?.id === imageId && afterErrorState.featureStatus === 'computing') {
            pollingTimeoutId = setTimeout(pollStatus, POLLING_INTERVAL * 2);
          } else if (afterErrorState.currentImage?.id === imageId && afterErrorState.featureStatus === 'computing') {
            set({ isFeatureLoading: false, featureStatus: 'error', error: 'Polling failed after multiple retries', isStartingFeatureOp: false });
          } else if (afterErrorState.isFeatureLoading) {
            set({ isFeatureLoading: false, isStartingFeatureOp: false });
          }
        }
      };

      // Kick off polling shortly after starting recompute
      setTimeout(pollStatus, POLLING_INTERVAL / 2);
    } catch (error) {
      console.error(`Error recomputing features for ${imageId}:`, error);
      const state = get();
      if (state.currentImage?.id === imageId) {
        set({ featureStatus: 'error', isFeatureLoading: false, error: error instanceof Error ? error.message : 'Failed to recompute features' });
      }
    } finally {
      const finalState = get();
      if (finalState.isStartingFeatureOp) set({ isStartingFeatureOp: false });
    }
  },

  // Navigation functions for previous/next image
  switchToPreviousImage: () => {
    const { availableImages, currentImage } = get();
    if (!availableImages || availableImages.length === 0 || !currentImage) return;
    
    const currentIndex = availableImages.findIndex(img => img.id === currentImage.id);
    if (currentIndex > 0) {
      const previousImage = availableImages[currentIndex - 1];
      set({ currentImage: previousImage });
    }
  },

  switchToNextImage: () => {
    const { availableImages, currentImage } = get();
    if (!availableImages || availableImages.length === 0 || !currentImage) return;
    
    const currentIndex = availableImages.findIndex(img => img.id === currentImage.id);
    if (currentIndex >= 0 && currentIndex < availableImages.length - 1) {
      const nextImage = availableImages[currentIndex + 1];
      set({ currentImage: nextImage });
    }
  },
}));

// Do not import other stores here to avoid circular dependencies and side effects in actions
