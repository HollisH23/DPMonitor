//
//  CameraView.swift
//  DPMonitor
//
//  UIViewRepresentable wrapper around AVCaptureVideoPreviewLayer.
//
//  The preview layer renders the capture session directly on the render
//  server — the frames never round-trip through our process, so the live
//  preview costs essentially nothing on top of the pose pipeline that is
//  already consuming the same session.
//

import AVFoundation
import SwiftUI

struct CameraView: UIViewRepresentable {

    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        if uiView.videoPreviewLayer.session !== session {
            uiView.videoPreviewLayer.session = session
        }
    }

    /// A UIView whose backing layer *is* the preview layer, so it resizes
    /// with the view automatically instead of needing manual frame syncing
    /// in `layoutSubviews`.
    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            // Safe: `layerClass` guarantees the type.
            layer as! AVCaptureVideoPreviewLayer
        }
    }
}
