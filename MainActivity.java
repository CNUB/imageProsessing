package com.example.myapplication;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_CODE = 100;
    private static final String TAG = "MainActivity";

    private ExecutorService cameraExecutor;
    private TFLiteObjectDetection tfliteModel;
    private BoundingBoxOverlay boundingBoxOverlay;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        boundingBoxOverlay = findViewById(R.id.bounding_box_overlay);

        // 카메라 권한 확인 및 요청
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_CODE);
        } else {
            startCamera();
        }

        // TFLite 모델 로드
        try {
            tfliteModel = new TFLiteObjectDetection(this, "yolov8n_float32.tflite");
            Log.d(TAG, "TFLite model loaded successfully.");
        } catch (Exception e) {
            Log.e(TAG, "Error loading TFLite model", e);
        }

        // 카메라 실행을 위한 Executor
        cameraExecutor = Executors.newSingleThreadExecutor();
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // Preview 설정
                PreviewView previewView = findViewById(R.id.preview_view);
                androidx.camera.core.Preview preview = new androidx.camera.core.Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // ImageAnalysis 설정
                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, image -> analyzeImage(image));

                // Camera Selector (후면 카메라 사용)
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                // 카메라 연결
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
                Log.d(TAG, "Camera started successfully.");

            } catch (Exception e) {
                Log.e(TAG, "Error starting CameraX", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void analyzeImage(ImageProxy imageProxy) {
        Bitmap bitmap = ImageUtils.imageProxyToBitmap(imageProxy);

        if (bitmap == null) {
            Log.e(TAG, "Failed to convert ImageProxy to Bitmap.");
        } else if (tfliteModel == null) {
            Log.e(TAG, "TFLite model is not loaded.");
        } else {
            Log.d(TAG, "Bitmap and TFLite model are ready for inference.");

            // 비트맵 크기 조정 (예: 640x640)
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true);

            try {
                // TFLite 모델을 사용하여 객체 탐지 수행
                List<float[]> detectedBoxes = tfliteModel.detectObjects(resizedBitmap);

                Log.d(TAG, "Detected " + detectedBoxes.size() + " objects.");
                for (float[] box : detectedBoxes) {
                    Log.d(TAG, "Box: [" + box[0] + ", " + box[1] + ", " + box[2] + ", " + box[3] + "]");
                }

                // BoundingBoxOverlay에 탐지된 박스 전달
                runOnUiThread(() -> {
                    Log.d(TAG, "Updating BoundingBoxOverlay with " + detectedBoxes.size() + " boxes.");
                    boundingBoxOverlay.setBoundingBoxes(detectedBoxes);
                });
            } catch (Exception e) {
                Log.e(TAG, "Error during object detection", e);
            }
        }

        imageProxy.close(); // 이미지 닫기
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Log.e(TAG, "Camera permission denied.");
                finish(); // 권한이 거부되면 앱 종료
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
        if (tfliteModel != null) {
            tfliteModel.close();
        }
    }
}