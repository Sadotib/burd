import 'package:flutter/material.dart';
import 'package:turd/utils/helper/helper.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:turd/widgets/label_card.dart';
import 'package:flutter/services.dart';

class LabelImagesPage extends StatelessWidget {
  final Map<String, String> labelImageMap;

  const LabelImagesPage({super.key, required this.labelImageMap});

  void _launchURL() async {
    final url = Uri.parse('https://www.kaggle.com/datasets/ichhadhari/indian-birds');
    if (await canLaunchUrl(url)) {
      await launchUrl(url, mode: LaunchMode.inAppBrowserView);
    } else {
      throw 'Could not launch $url';
    }
  }

  @override
  Widget build(BuildContext context) {
    
    return Scaffold(
      appBar: AppBar(
        title: Text(
          "Dataset",
          style: TextStyle(
            color: HelperFunction.isDark(context)? Colors.white : Colors.black,
            fontSize: 40,
            fontFamily: 'MyFont',
            
          ),
        ),
        actions: [
          IconButton(
            onPressed: _launchURL,
            icon: Icon(Icons.open_in_new_rounded, size: 30),
          ),
        ],
        systemOverlayStyle: SystemUiOverlayStyle(
          statusBarColor: HelperFunction.isDark(context)? Colors.black : Colors.white, // Now this will actually work
          statusBarIconBrightness: HelperFunction.isDark(context)? Brightness.light : Brightness.dark,
        ),
      ),
      body: ListView.builder(
        // itemCount: labelImageMap.length,
        itemCount: labelImageMap.length,
        itemBuilder: (context, index) {
          final label = labelImageMap.keys.elementAt(index);
          final imagePath = labelImageMap[label];
          print(imagePath);

          // return Card(
          //   shape: RoundedRectangleBorder(
          //     borderRadius: BorderRadius.circular(20),
          //   ),
          //   margin: const EdgeInsets.all(15),
          //   child: ListTile(
          //     // leading: Image.asset(
          //     //   imagePath!,
          //     //   width: 60,
          //     //   height: 60,

          //     //   fit: BoxFit.cover,
          //     //   errorBuilder: (_, __, ___) => const Icon(Icons.broken_image),
          //     // ),
          //     title: Text(label, style: TextStyle(color: Colors.white)),
          //   ),
          return LabelCard(label: label, imagePath: imagePath!);
          
        },
      ),
    );
  }
}
