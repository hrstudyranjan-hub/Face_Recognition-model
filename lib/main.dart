import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'ui/camera_view.dart';
import 'services/ml_vision_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Lock orientation to portrait
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  
  // Initialize the ML Vision Service globally or via Provider
  final mlVisionService = MLVisionService();
  await mlVisionService.initializeModels();

  runApp(NexalayaApp(mlVisionService: mlVisionService));
}

class NexalayaApp extends StatelessWidget {
  final MLVisionService mlVisionService;

  const NexalayaApp({super.key, required this.mlVisionService});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Nexalaya 2.0',
      theme: ThemeData(
        brightness: Brightness.dark,
        primarySwatch: Colors.blue,
      ),
      home: DashboardScreen(mlVisionService: mlVisionService),
      debugShowCheckedModeBanner: false,
    );
  }
}

class DashboardScreen extends StatelessWidget {
  final MLVisionService mlVisionService;

  const DashboardScreen({super.key, required this.mlVisionService});

  Future<void> initiateFaceWorkflow(BuildContext context) async {
    const storage = FlutterSecureStorage();
    
    String? storedEmbedding;
    try {
      // Query secure storage for an existing embedding
      storedEmbedding = await storage.read(key: 'student_face_embedding');
    } catch (e, stackTrace) {
      debugPrint('Secure Storage Error [initiateFaceWorkflow Read]: $e');
      debugPrint('Stack Trace: $stackTrace');
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error initializing Face Auth Mode: Storage Read Failed'),
            backgroundColor: Colors.redAccent,
          ),
        );
      }
      return; // Exit if storage read fails entirely
    }
    
    // Determine mode based on whether the embedding exists
    final FaceAuthMode mode = (storedEmbedding == null) 
        ? FaceAuthMode.register 
        : FaceAuthMode.verify;

    if (context.mounted) {
      // Await the result of the workflow to handle success/failure down the line
      final result = await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => CameraView(
            mlVisionService: mlVisionService,
            mode: mode,
          ),
        ),
      );
      
      // Handle result
      if (result != null && context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(result.toString())),
        );
      }
    }
  }

  Future<void> deleteRegisteredFace(BuildContext context) async {
    const storage = FlutterSecureStorage();
    try {
      await storage.delete(key: 'student_face_embedding');
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Registered face deleted successfully!')),
        );
      }
    } catch (e, stackTrace) {
      debugPrint('Secure Storage Error [deleteRegisteredFace]: $e');
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Error deleting registered face.'),
            backgroundColor: Colors.redAccent,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Dashboard')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () => initiateFaceWorkflow(context),
              child: const Text('Face Recognition / Registration'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => deleteRegisteredFace(context),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.redAccent,
                foregroundColor: Colors.white,
              ),
              child: const Text('Delete Registered Face'),
            ),
          ],
        ),
      ),
    );
  }
}
