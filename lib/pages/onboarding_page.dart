import 'package:turd/utils/controllers/onboarding_controller.dart';
import 'package:turd/utils/helper/helper.dart';
import 'package:flutter/material.dart';
import 'package:smooth_page_indicator/smooth_page_indicator.dart';
import 'package:get/get.dart';

class OnboardingScreen extends StatelessWidget {
  const OnboardingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = Get.put(OnboardingController());

    return Scaffold(
      body: Stack(
        children: [
          PageView(
            controller: controller.pageController,
            onPageChanged: controller.updatePageIndicator,
            children: [
              OnboardingPage(
                image: "assets/Scene-1.gif",
                title: "Welcome to",
                subtitle: "Burd",
              ),
              OnboardingPage(
                image: "assets/Frame 1.png",
                title: "Brought to you by",
                subtitle: "BIN Labs",
              ),
              OnboardingPage(
                image: "assets/Frame 1 (1).png",
                title: "Find birds easily",
                subtitle: "Try it out",
              ),
            ],
          ),
          const OnboardingSkip(),
          const OnboardingDot(),
          const OnboardingNext(),
        ],
      ),
    );
  }
}

class OnboardingNext extends StatelessWidget {
  const OnboardingNext({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      right: MediaQuery.of(context).size.width*0.05,
      bottom: MediaQuery.of(context).size.height * 0.045,
      child: FilledButton(
        onPressed: ()=> OnboardingController.instance.nextPage(), 
        style: ButtonStyle(
          shape: WidgetStateProperty.all<CircleBorder>(CircleBorder()),
        ),
        child: Icon(Icons.arrow_forward_rounded,size: 30,),
      ),
    );
  }
}

class OnboardingDot extends StatelessWidget {
  const OnboardingDot({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    final controller = OnboardingController.instance;
    return Positioned(
      bottom: MediaQuery.of(context).size.height * 0.06,
      left: MediaQuery.of(context).size.width * 0.05,
      child: SmoothPageIndicator(
        effect: SlideEffect(
          activeDotColor: Color(0xFF1F4BEA),
          dotHeight: 15,
          dotColor: HelperFunction.isDark(context)?Color(0xFFFFFFFF) : Color(0xFF000000),
          paintStyle: PaintingStyle.stroke,
        ),
        controller: controller.pageController, 
        onDotClicked: controller.dotNavigationClick,
        count: 3),
    );
  }
}

class OnboardingSkip extends StatelessWidget {
  const OnboardingSkip({super.key});

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: MediaQuery.of(context).size.height * 0.04,
      right: MediaQuery.of(context).size.width * 0.05,
      child: TextButton(
        onPressed: () => OnboardingController.instance.skipPage(),
        style: ButtonStyle(
          foregroundColor:
              HelperFunction.isDark(context)
                  ? WidgetStateProperty.all<Color>(Color(0xFF1F4BEA))
                  : WidgetStateProperty.all<Color>(Color(0xFF1F4BEA)),
          overlayColor: WidgetStateProperty.all<Color>(
            Color.fromARGB(92, 31, 75, 234),
          ),
          
        ),
        child: Text("Skip"),
      ),
    );
  }
}

class OnboardingPage extends StatelessWidget {
  const OnboardingPage({
    super.key,
    required this.image,
    required this.title,
    required this.subtitle,
  });
  final String image, title, subtitle;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Image(
          image: AssetImage(image),
          width: MediaQuery.of(context).size.width * 0.9,
          height: MediaQuery.of(context).size.height * 0.6,
        ),
        Text(
          title,
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.headlineMedium,
        ),
        const SizedBox(height: 10),
        Text(
          subtitle,
          style: Theme.of(context).textTheme.bodyMedium,
          textAlign: TextAlign.center,
        ),
      ],
    );
  }
}
