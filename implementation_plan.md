# UI/UX Refinement: Optimize Webcam Field of View (FOV)

## Goal Description
In the Live Monitoring View, the video feed currently appears cropped and zoomed in due to the `object-fit: cover` styling. This restricts the patient's visual Field of View (FOV), making it difficult to keep their full body in the frame for accurate MediaPipe skeleton tracking. 

We will modify the frontend to display the entire, uncropped raw video feed (using `object-fit: contain` with black-bar letterboxing) and explicitly request a widescreen, high-resolution (1080p, 16:9) camera stream from the patient's device via `getUserMedia` to maximize physical capture space.

## User Review Required

> [!IMPORTANT]
> **Custom Camera Wrapper Implementation**: Since the `@mediapipe/camera_utils` `Camera` constructor only accepts primitive values for width and height and maps them to a restricted options object internally, we cannot pass the required `ideal` object constraints payload through it. We will bypass the `@mediapipe/camera_utils` library in `usePoseDetection.js` and instantiate standard `navigator.mediaDevices.getUserMedia` constraints and custom frame loops directly.

> [!NOTE]
> **Letterboxing**: Transitioning from `object-fit: cover` to `object-fit: contain` will display the full widescreen camera frame, but may introduce horizontal or vertical letterboxes depending on the container's proportions. This ensures that keypoint detection functions optimally on the entire body.

## Proposed Changes

---

### Component: Frontend UI & Pose Detection

#### [MODIFY] [usePoseDetection.js](file:///Users/houxiqing/Documents/USYD/ProfJinman/DPMonitor/frontend/src/hooks/usePoseDetection.js)
- Replace the usage of `@mediapipe/camera_utils` `Camera` helper class with standard `navigator.mediaDevices.getUserMedia`.
- Configure the stream using the following constraint standard to enforce widescreen 1080p user-facing capture:
  ```javascript
  const constraints = {
    video: {
      facingMode: "user",
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      aspectRatio: { ideal: 1.7777777778 } // 16:9 aspect ratio
    },
    audio: false
  };
  ```
- Implement a custom frame-pump loop using `requestAnimationFrame` that continuously invokes `poseRef.current.send({ image: videoRef.current })` on each active frame.
- Retain the track-stopping logic inside `stopCamera` using the tracks returned by the media stream.

#### [MODIFY] [global.css](file:///Users/houxiqing/Documents/USYD/ProfJinman/DPMonitor/frontend/src/styles/global.css)
- Find `.viewport video` and `.viewport canvas.overlay` (lines 324–331) and update `object-fit` from `cover` to `contain`.
- Verify that mirroring styles (`.viewport video.mirrored` and `.overlay-mirror` which use `transform: scaleX(-1)`) remain active and unaffected.

---

## Verification Plan

### Manual Verification
1. **Camera Constraints & Resolution Check**:
   - Launch the application locally and navigate to the Live Monitor view.
   - Inspect the video element's videoWidth and videoHeight via browser Developer Tools to confirm the device is streaming at 16:9 widescreen format (ideally 1920x1080 if supported by hardware, or falling back to the device's closest 16:9 configuration).
2. **Visual FOV & Layout Integrity**:
   - Confirm that the video feed shows the complete uncropped camera view with zero cropping.
   - Verify that the layout does not break and stays centered within the viewport container (surrounded by letterboxing if the viewport has different dimensions).
   - Ensure the canvas skeleton overlay lines up perfectly with the body landmarks shown in the video.
3. **Horizontal Mirroring Check**:
   - Lift your right arm and ensure it shows up on the left side of the screen (mimicking a mirror).
   - Verify that the canvas skeleton overlay follows the mirrored video feed exactly.
