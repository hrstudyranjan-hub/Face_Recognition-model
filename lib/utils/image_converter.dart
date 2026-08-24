import 'dart:typed_data';
import 'package:camera/camera.dart';

class ImageConverter {
  /// Converts a CameraImage to a normalized flat Float32List RGB tensor.
  /// Standard normalized scale: (pixel - 127.5) / 128.0 (matches fixed_image_standardization)
  static Float32List convertCameraImageToTensor(
    CameraImage image, 
    bool isNCHW,
  ) {
    if (image.format.group == ImageFormatGroup.yuv420) {
      return _convertYUV420(image, isNCHW);
    } else if (image.format.group == ImageFormatGroup.bgra8888) {
      return _convertBGRA8888(image, isNCHW);
    } else {
      throw Exception('Unsupported image format: ${image.format.group}');
    }
  }

  static Float32List _convertYUV420(CameraImage image, bool isNCHW) {
    final int width = image.width;
    final int height = image.height;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

    // Total elements: 3 channels * width * height
    final Float32List tensor = Float32List(3 * width * height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex =
            uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
        final int index = y * image.planes[0].bytesPerRow + x;

        final int yp = image.planes[0].bytes[index];
        final int up = image.planes[1].bytes[uvIndex];
        final int vp = image.planes[2].bytes[uvIndex];

        // YUV to RGB math
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 352 / 1024 - vp * 731 / 1024 + 135).round().clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 226).round().clamp(0, 255);

        // Normalize: (val - 127.5) / 128.0
        double normR = (r - 127.5) / 128.0;
        double normG = (g - 127.5) / 128.0;
        double normB = (b - 127.5) / 128.0;

        if (isNCHW) {
          // NCHW Layout: Channels -> Height -> Width
          int cStride = width * height;
          tensor[0 * cStride + y * width + x] = normR;
          tensor[1 * cStride + y * width + x] = normG;
          tensor[2 * cStride + y * width + x] = normB;
        } else {
          // NHWC Layout: Height -> Width -> Channels
          int pixelIndex = (y * width + x) * 3;
          tensor[pixelIndex] = normR;
          tensor[pixelIndex + 1] = normG;
          tensor[pixelIndex + 2] = normB;
        }
      }
    }
    return tensor;
  }

  static Float32List _convertBGRA8888(CameraImage image, bool isNCHW) {
    final int width = image.width;
    final int height = image.height;
    final Float32List tensor = Float32List(3 * width * height);
    final Uint8List bytes = image.planes[0].bytes;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int index = (y * width + x) * 4;

        final int b = bytes[index];
        final int g = bytes[index + 1];
        final int r = bytes[index + 2];

        double normR = (r - 127.5) / 128.0;
        double normG = (g - 127.5) / 128.0;
        double normB = (b - 127.5) / 128.0;

        if (isNCHW) {
          int cStride = width * height;
          tensor[0 * cStride + y * width + x] = normR;
          tensor[1 * cStride + y * width + x] = normG;
          tensor[2 * cStride + y * width + x] = normB;
        } else {
          int pixelIndex = (y * width + x) * 3;
          tensor[pixelIndex] = normR;
          tensor[pixelIndex + 1] = normG;
          tensor[pixelIndex + 2] = normB;
        }
      }
    }
    return tensor;
  }
}
