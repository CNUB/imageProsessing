package com.example.myapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class TFLiteObjectDetection {

    private static final String TAG = "TFLiteObjectDetection";
    private Interpreter interpreter;

    public TFLiteObjectDetection(Context context, String modelPath) throws IOException {
        try {
            interpreter = new Interpreter(loadModelFile(context, modelPath));
            Log.d(TAG, "TFLite model loaded successfully: " + modelPath);
        } catch (Exception e) {
            Log.e(TAG, "Error loading TFLite model", e);
            throw e;
        }
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        FileInputStream inputStream = context.getAssets().openFd(modelPath).createInputStream();
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = context.getAssets().openFd(modelPath).getStartOffset();
        long declaredLength = context.getAssets().openFd(modelPath).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
            Log.d(TAG, "TFLite interpreter closed successfully.");
        }
    }

    /**
     * 객체 탐지를 수행하는 메서드
     *
     * @param bitmap 입력 이미지 (Bitmap 형식)
     * @return 탐지된 바운딩 박스 리스트
     */
    public List<float[]> detectObjects(Bitmap bitmap) {
        int inputSize = 640; // 모델 입력 크기
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);

        // 입력 데이터 준비
        float[][][][] input = new float[1][inputSize][inputSize][3];
        for (int y = 0; y < inputSize; y++) {
            for (int x = 0; x < inputSize; x++) {
                int pixel = resizedBitmap.getPixel(x, y);
                input[0][y][x][0] = (pixel >> 16 & 0xFF) / 255.0f; // R
                input[0][y][x][1] = (pixel >> 8 & 0xFF) / 255.0f;  // G
                input[0][y][x][2] = (pixel & 0xFF) / 255.0f;       // B
            }
        }

        // 모델의 출력 형상이 [1, 84, 8400]임
        float[][][] output = new float[1][84][8400]; // 예: [1, 84, 8400]

        // 추론 수행
        interpreter.run(input, output);

        // 바운딩 박스와 신뢰도 추출
        List<float[]> boundingBoxes = new ArrayList<>();
        for (int i = 0; i < 8400; i++) { // 8400까지 반복
            // i번째 예측 정보에 접근
            float[] prediction = output[0][i % 84]; // 0부터 83까지의 인덱스 사용
            float confidence = prediction[4]; // 신뢰도 (예: 0.5 이상)

            if (confidence > 0.3) { // 신뢰도가 50% 이상인 경우만 선택
                float xCenter = prediction[0];
                float yCenter = prediction[1];
                float width = prediction[2];
                float height = prediction[3];

                // 바운딩 박스 좌표 계산
                float x1 = (xCenter - width / 2) * bitmap.getWidth();
                float y1 = (yCenter - height / 2) * bitmap.getHeight();
                float x2 = (xCenter + width / 2) * bitmap.getWidth();
                float y2 = (yCenter + height / 2) * bitmap.getHeight();

                boundingBoxes.add(new float[]{x1, y1, x2, y2});
                Log.d(TAG, "Confidence: " + confidence + ", Box: [" + x1 + ", " + y1 + ", " + x2 + ", " + y2 + "]");
            }
        }

        return boundingBoxes;
    }

}
