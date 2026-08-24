import 'package:flutter/material.dart';

class FaceResult {
  final Rect boundingBox;
  final bool isReal;
  final double spoofConfidence;

  FaceResult({
    required this.boundingBox,
    required this.isReal,
    required this.spoofConfidence,
  });
}

class BoundingBoxPainter extends CustomPainter {
  final List<FaceResult> faces;
  final Size imageSize;
  final InputImageRotation rotation;

  BoundingBoxPainter(this.faces, this.imageSize, this.rotation);

  @override
  void paint(Canvas canvas, Size size) {
    final double scaleX = size.width / imageSize.width;
    final double scaleY = size.height / imageSize.height;

    final Paint boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    for (final face in faces) {
      // Choose color based on anti-spoofing result
      boxPaint.color = face.isReal ? Colors.green : Colors.red;

      // Scale bounding box to match the screen size
      final Rect rect = Rect.fromLTRB(
        face.boundingBox.left * scaleX,
        face.boundingBox.top * scaleY,
        face.boundingBox.right * scaleX,
        face.boundingBox.bottom * scaleY,
      );

      canvas.drawRect(rect, boxPaint);

      // Draw the Anti-Spoofing Text
      final textSpan = TextSpan(
        text: face.isReal 
            ? 'Real (${(face.spoofConfidence * 100).toStringAsFixed(1)}%)'
            : 'Spoof (${(face.spoofConfidence * 100).toStringAsFixed(1)}%)',
        style: TextStyle(
          color: boxPaint.color,
          fontSize: 18,
          fontWeight: FontWeight.bold,
          backgroundColor: Colors.black54,
        ),
      );

      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(canvas, Offset(rect.left, rect.top - 25));
    }
  }

  @override
  bool shouldRepaint(covariant BoundingBoxPainter oldDelegate) {
    return oldDelegate.faces != faces || oldDelegate.imageSize != imageSize;
  }
}

// Temporary enum to hold rotation if using camera image directly
enum InputImageRotation {
  rotation0deg,
  rotation90deg,
  rotation180deg,
  rotation270deg,
}
