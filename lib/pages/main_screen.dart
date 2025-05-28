// lib/pages/main_screen.dart
import 'package:flutter/material.dart';
import 'home_page.dart';
import 'settings_screen.dart';
import 'package:flutter_native_splash/flutter_native_splash.dart';
// import 'package:flutter_tflite/flutter_tflite.dart';
import 'package:turd/pages/more_screen.dart';
import 'package:flutter/services.dart';
import 'package:turd/utils/helper/label_image_helper.dart';


class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final PageController controller = PageController(initialPage: 1);
  int _currentIndex = 1;
  Map<String, String>? labelImageMap;

  // @override
  // void dispose() {
  //   interpreter.close();
  // }

  @override
  void initState() {
    super.initState();
    print("inside home state init");

    SystemChrome.setSystemUIOverlayStyle(
      SystemUiOverlayStyle(
        statusBarColor: Colors.green, // 🔷 Your desired status bar color
        statusBarIconBrightness: Brightness.light, // For icon contrast
      ),
    );
    _initializeApp();
  }

  Future<void> _initializeApp() async {
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      FlutterNativeSplash.remove();
      // await tfLteInit();
      final map = await LabelImageHelper.getLabelImageMap();
      setState(() {
        labelImageMap = map;
      });
    }); // FlutterNativeSplash.remove();
  }

  void _goToPage(int index) {
    setState(() {
      _currentIndex = index;
    });
    controller.animateToPage(
      index,
      duration: const Duration(milliseconds: 200),
      curve: Curves.easeInOut,
    );
  }

  @override
  Widget build(BuildContext context) {
    // bool isDark = Theme.of(context).brightness == Brightness.dark;

    if (labelImageMap == null) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      // appBar: AppBar(
      //   leading: IconButton(
      //     icon: const Icon(Icons.home),
      //     onPressed: () => _goToPage(0),
      //   ),
      //   title: const Text('BURD'),
      //   actions: [
      //     IconButton(
      //       icon: const Icon(Icons.settings),
      //       onPressed: () => _goToPage(1),
      //     ),
      //   ],
      // ),
      bottomNavigationBar: BottomAppBar(
        height: MediaQuery.of(context).size.height * 0.08,

        // color: Theme.of(context).appBarTheme.backgroundColor,
        // color: Colors.green,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            IconButton(
              icon: Icon(
                Icons.settings,
                color: _currentIndex == 0 ? Color(0xFF1F4BEA) : Colors.grey,
                size: _currentIndex == 0 ? 40 : 25,
              ),
              onPressed: () => _goToPage(0),
              // onPressed: () {},
            ),
            IconButton(
              icon: Icon(
                Icons.home,
                color: _currentIndex == 1 ? Color(0xFF1F4BEA) : Colors.grey,
                size: _currentIndex == 1 ? 40 : 25,
              ),
              onPressed: () => _goToPage(1),
              // onPressed: () {},
            ),
            IconButton(
              onPressed: () => _goToPage(2),
              icon: Icon(
                Icons.data_array,
                color: _currentIndex == 2 ? Color(0xFF1F4BEA) : Colors.grey,
                size: _currentIndex == 2 ? 40 : 25,
              ),
            ),
          ],
        ),
      ),

      body: PageView(
        controller: controller,
        onPageChanged: (index) => setState(() => _currentIndex = index),
        children: [
          SettingsPage(),
          HomePage(),
          LabelImagesPage(labelImageMap: labelImageMap!),
        ],
      ),
    );
  }
}

// class OnboardingDot extends StatelessWidget {
//   const OnboardingDot({
//     super.key,
//   });

//   @override
//   Widget build(BuildContext context) {
//     final controller = controller.instance;
//     return Positioned(
//       bottom: MediaQuery.of(context).size.height * 0.06,
//       left: MediaQuery.of(context).size.width * 0.05,
//       child: SmoothPageIndicator(
//         effect: SlideEffect(
//           activeDotColor: Color(0xFF1F4BEA),
//           dotHeight: 15,
//           dotColor: HelperFunction.isDark(context)?Color(0xFFFFFFFF) : Color(0xFF000000),
//           paintStyle: PaintingStyle.stroke,
//         ),
//         controller: controller.pageController,
//         onDotClicked: controller.dotNavigationClick,
//         count: 3),
//     );
//   }
// }



// Future<void> tfLteInit() async {
//   print("inside loading model");
//   final interpreter = await Interpreter.fromAsset(
//     'assets/models/bird_model_float32.tflite',
//   );
//   interpreter.allocateTensors();
//   final inputShape = interpreter.getInputTensor(0).shape; // [1, 224, 224, 3]
//   final outputShape = interpreter.getOutputTensor(0).shape;
//   final outputType = interpreter.getOutputTensor(0).type;
  
  
// }

