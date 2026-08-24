import 'dart:isolate';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import '../utils/image_converter.dart';

/// Data class to send from Main Thread to Isolate
class IsolateData {
  final CameraImage cameraImage;
  final bool isNCHW;

  IsolateData({
    required this.cameraImage,
    required this.isNCHW,
  });
}

class IsolateUtils {
  static const String DEBUG_NAME = 'MLVisionIsolate';

  Isolate? _isolate;
  final ReceivePort _receivePort = ReceivePort();
  SendPort? _sendPort;

  SendPort? get sendPort => _sendPort;

  Future<void> start() async {
    _isolate = await Isolate.spawn<SendPort>(
      _entryPoint,
      _receivePort.sendPort,
      debugName: DEBUG_NAME,
    );

    // Wait for the isolate to send its SendPort back to the main thread
    _sendPort = await _receivePort.first as SendPort;
  }

  static void _entryPoint(SendPort sendPort) async {
    final port = ReceivePort();
    sendPort.send(port.sendPort);

    await for (final IsolateData data in port) {
      try {
        final Float32List tensor = ImageConverter.convertCameraImageToTensor(
          data.cameraImage,
          data.isNCHW,
        );
        sendPort.send(tensor);
      } catch (e) {
        sendPort.send(e);
      }
    }
  }

  void dispose() {
    _receivePort.close();
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
  }
}
