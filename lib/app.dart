import 'package:flutter/material.dart';
import 'package:turd/pages/main_screen.dart';
import 'package:turd/pages/onboarding_page.dart';
import 'package:turd/utils/theme/theme.dart';

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.

  
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Turd',
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: ThemeMode.system,
      home: MainScreen(),
    );
  }
}