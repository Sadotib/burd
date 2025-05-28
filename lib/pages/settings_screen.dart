import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:turd/utils/helper/helper.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {

    SystemChrome.setSystemUIOverlayStyle(
      SystemUiOverlayStyle(
        statusBarColor: HelperFunction.isDark(context)? Colors.black : Colors.white, // your desired color
        statusBarIconBrightness:
            HelperFunction.isDark(context)? Brightness.light : Brightness.dark, // white icons for dark background
      ),
    );
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          color: HelperFunction.isDark(context)? Colors.black : Colors.white,
          // gradient: LinearGradient(
          //   colors: [Color(0xFFFFFFFF), Color(0xFF1F4BEA)],
          //   begin: Alignment.topCenter,
          //   end: Alignment.bottomCenter,

          // ),
        ),
        child: Center(
          child: Text(
          "IN DEVELOPMENT",

          ),
        ),
      ),
    );
  
    
  }
}