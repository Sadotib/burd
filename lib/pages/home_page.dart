import 'package:turd/pages/info_screen.dart';
import 'package:turd/utils/common/tflite_model_data.dart';
import 'package:turd/utils/helper/helper.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:turd/utils/theme/iconbutton_theme.dart';
import 'package:turd/utils/theme/theme.dart';
import 'dart:io';
import 'package:simple_gradient_text/simple_gradient_text.dart';
import 'package:turd/widgets/image_card.dart';
import 'package:turd/utils/api/api_call.dart';
import 'package:turd/utils/helper/bird_result.dart';
import 'package:turd/utils/helper/crop_image.dart';
import 'package:flutter/services.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<StatefulWidget> createState() {
    return HomePageState();
  }
}

class HomePageState extends State<HomePage> {
  File? filePath;
  String label = '';
  double conf = 0.0;
  bool isLoading = false;

  pickImageGallery() async {
    final ImagePicker picker = ImagePicker();
    // Pick an image.
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);

    if (image == null) return;

    final File? cropped = await cropImage(image);
    if (cropped == null) return;

    // var imageMap = File(image.path);

    setState(() {
      filePath = cropped;
      imageFile = cropped;
    });
    // classifyImage(image.path);
    print(image.path);

    // imageFile = imageMap;
  }

  pickImageCamera() async {
    final ImagePicker picker = ImagePicker();
    // Pick an image.
    final XFile? image = await picker.pickImage(source: ImageSource.camera);

    if (image == null) return;

    // var imageMap = File(image.path);

    final File? cropped = await cropImage(image);
    if (cropped == null) return;

    setState(() {
      filePath = cropped;
      imageFile = cropped;
    });

    // imageFile = imageMap;

    print("hello" + image.path);
  }

  void _clearCard() {
    setState(() {
      filePath = null;
      label = '';
      conf = 0.0;
    });
  }

  Future<dynamic> _openInfo() {
    return showModalBottomSheet(
      context: context,
      builder: (ctx) => InfoScreen(),
      isScrollControlled: false,
      enableDrag: true,
      isDismissible: true,
      useSafeArea: true,
      showDragHandle: true,
    );
  }

  @override
  Widget build(BuildContext context) {
    // bool isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      appBar: AppBar(
        toolbarHeight: MediaQuery.of(context).size.height * 0.05,
        title: GradientText(
          "BURD",

          style: TextStyle(
            fontSize: 40,
            fontWeight: FontWeight.w200,
            fontFamily: 'MyFont',
          ),

          colors:
              HelperFunction.isDark(context)
                  ? [
                    Color(0xFF1F4BEA),
                    Color.fromRGBO(81, 112, 224, 1),
                    Color.fromARGB(255, 255, 255, 255),
                  ]
                  : [
                    Color(0xFF1F4BEA),
                    Color.fromRGBO(81, 112, 224, 1),
                    Color.fromARGB(255, 0, 0, 0),
                  ],
        ),
        // actions: [
        //   IconButton(
        //     onPressed: () {
        //       _goToPage(0);
        //     },
        //     icon: Icon(Icons.home),
        //     color:
        //         HelperFunction.isDark(context) && _currentIndex == 0
        //             ? Color(0xFF1F4BEA)
        //             : Colors.white,
        //   ),
        // ],
        actions: [
          IconButton(
            onPressed: _openInfo,

            // Navigator.push(
            //       context,
            //       MaterialPageRoute(builder: (context) => SettingsPage()),
            // );
            icon: Icon(Icons.info, size: 30,color: HelperFunction.isDark(context) ? Colors.white : Colors.black,),
            // color: HelperFunction.isDark(context) ? Colors.white : Colors.black,

            // ? Color(0xFF1F4BEA)
            // : Color(0xFF1F4BEA),
          ),
        ],
        centerTitle: true,
        systemOverlayStyle: SystemUiOverlayStyle(
          statusBarColor: HelperFunction.isDark(context)? Colors.black : Colors.white, // Now this will actually work
          statusBarIconBrightness: HelperFunction.isDark(context)? Brightness.light : Brightness.dark,
        ),
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            children: [
              const SizedBox(height: 12),
              ImageCard(filePath: filePath),
              const SizedBox(height: 8),
              _uploadButtons(),
              const SizedBox(height: 8),
              if (filePath != null) _clearButton(),
              if (filePath != null && label.isNotEmpty) _predictionBox(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _uploadButtons() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.start,
      mainAxisSize: MainAxisSize.max,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(width: MediaQuery.of(context).size.width * 0.05),
        Container(
          width: MediaQuery.of(context).size.width * 0.13,
          height: MediaQuery.of(context).size.height * 0.06,

          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(50),
            color: HelperFunction.isDark(context)? Colors.white : Colors.black,
          ),
          child: IconButtonTheme(
            // data:
            //     HelperFunction.isDark(context)
            //         ? AppTheme.darkTheme.filledButtonTheme
            //         : AppTheme.lightTheme.filledButtonTheme,
            data:
                HelperFunction.isDark(context)
                    ? AppIconButtonTheme.darkIconButtonTheme
                    : AppIconButtonTheme.lightIconButtonTheme,

            child: IconButton(
              icon: Icon(Icons.camera_alt),
              onPressed: () {
                setState(() {
                  filePath = null;
                  label = '';
                  conf = 0.0;
                });
                pickImageCamera();
              },
            ),
          ),
        ),
        SizedBox(width: MediaQuery.of(context).size.width * 0.03),
        Container(
          width: MediaQuery.of(context).size.width * 0.13,
          height: MediaQuery.of(context).size.height * 0.06,

          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(50),
            border: Border.all(width: 2),
            color: HelperFunction.isDark(context)? Colors.white : Colors.black,
          ),
          child: IconButtonTheme(
            // data:
            //     HelperFunction.isDark(context)
            //         ? AppTheme.darkTheme.filledButtonTheme
            //         : AppTheme.lightTheme.filledButtonTheme,
            data:
                HelperFunction.isDark(context)
                    ? AppIconButtonTheme.darkIconButtonTheme
                    : AppIconButtonTheme.lightIconButtonTheme,

            child: IconButton(
              icon: Icon(Icons.add_photo_alternate_rounded),
              onPressed: () {
                setState(() {
                  filePath = null;
                  label = '';
                  conf = 0.0;
                });
                pickImageGallery();
              },
            ),
          ),
        ),
        SizedBox(width: MediaQuery.of(context).size.width * 0.03),
        if (filePath != null)
          Container(
            width: MediaQuery.of(context).size.width * 0.58,

            height: MediaQuery.of(context).size.height * 0.06,
            child: FilledButtonTheme(
              data:
                  HelperFunction.isDark(context)
                      ? AppTheme.darkTheme.filledButtonTheme
                      : AppTheme.lightTheme.filledButtonTheme,
              child: FilledButton(
                onPressed: () async {
                  // try {
                  //   await predictBird(imageFile);
                  //   setState(() {
                  //     label = BirdResult().prediction;
                  //     conf = BirdResult().confidence;
                  //   });
                  // } catch (e) {
                  //   print("Prediction error: $e");
                  // }

                  setState(() {
                    isLoading = true;
                  });
                  try {
                    await predictBird(imageFile);
                    setState(() {
                      label = BirdResult().prediction;
                      conf = BirdResult().confidence;
                    });
                  } catch (e) {
                    print("Prediction error: $e");
                  } finally {
                    setState(() {
                      isLoading = false;
                    });
                  }
                },
                child: const Text("Detect"),
              ),
            ),
          ),
      ],
    );
  }

  Widget _clearButton() {
    return Container(
      width: MediaQuery.of(context).size.width * 0.9,
      height: MediaQuery.of(context).size.height * 0.06,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(50),
        border: Border.all(width: 2),
        color: Color(0xFFFFFFFF),
      ),
      child: IconButtonTheme(
        data:
            HelperFunction.isDark(context)
                ? AppIconButtonTheme.darkIconButtonTheme
                : AppIconButtonTheme.lightIconButtonTheme,

        child: IconButton(
          icon: Icon(Icons.close_sharp),
          onPressed: () {
            _clearCard();
          },
        ),
      ),
    );
  }

  Widget _predictionBox() {
    return Container(
      width: MediaQuery.of(context).size.width * 0.9,
      padding: EdgeInsets.all(16),
      margin: EdgeInsets.only(top: 10),
      decoration: BoxDecoration(
        color:
            HelperFunction.isDark(context)
                ? Colors.grey[900]
                : Colors.grey[200],
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.blueAccent, width: 1.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            "Prediction: $label",
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 4),
          Text(
            "Confidence: ${conf.toStringAsFixed(2)}%",
            style: TextStyle(fontSize: 16),
          ),
        ],
      ),
    );
  }
}
