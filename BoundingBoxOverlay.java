package com.example.myapplication;
import android.util.Log;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class BoundingBoxOverlay extends View {

    private final Paint boxPaint = new Paint();
    private List<float[]> boundingBoxes = new ArrayList<>();

    public BoundingBoxOverlay(Context context, AttributeSet attrs) {
        super(context, attrs);

        // 박스 스타일 설정
        boxPaint.setColor(Color.RED);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(8.0f);
    }

    // 바운딩 박스 업데이트
    public void setBoundingBoxes(List<float[]> boxes) {
        this.boundingBoxes = boxes;
        invalidate(); // 뷰를 다시 그립니다.
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        Log.d("BoundingBoxOverlay", "onDraw called with " + boundingBoxes.size() + " boxes.");
        for (float[] box : boundingBoxes) {
            Log.d("BoundingBoxOverlay", "Drawing box: [" + box[0] + ", " + box[1] + ", " + box[2] + ", " + box[3] + "]");
            canvas.drawRect(box[0], box[1], box[2], box[3], boxPaint);
        }
    }
}
